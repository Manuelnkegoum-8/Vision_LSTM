import numpy as np
import random,json
import torch
from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim
from models.xlstm import *
from utils.helpers import *
from utils.trainer import *
from torch.cuda.amp import GradScaler, autocast
from colorama import Fore, Style
import os
import torch.optim.lr_scheduler as lr_schedule
from utils.lr_scheduler import CosineWithLinearWarmup

import argparse
from tqdm import trange
from math import ceil
parser = argparse.ArgumentParser(description='IMAGE CLASSIFICATION WITH VISION LSTM')

# Data args
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--freq', default=10, type=int, metavar='N', help='log frequency (by iteration)')
parser.add_argument('--dataset', default='mnist', type=str, metavar='N', help='dataset')


# Model parameters
parser.add_argument('--qk_size', default=4, type=int, help='number of heads in m_lstm cell')
parser.add_argument('--classes', default=10, type=int, help='number of classes')
parser.add_argument('--width', default=224, type=int, help='image width')
parser.add_argument('--height', default=224, type=int, help='image height')
parser.add_argument('--channels', default=3, type=int, help='number of channels in image')
parser.add_argument('--patch_size', default=16, type=int, help='patch size')
parser.add_argument('--m_blocks', default=4, type=int, help='number of mlstm blocks')
parser.add_argument('--s_blocks', default=4, type=int, help='number of slstm blocks')
parser.add_argument('--dim', default=192, type=int, help='embedding dim of patch')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
parser.add_argument('--classif_mode', default='bilateral_avg', type=str, help='classif mode')


# Optimization hyperparams
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--amp', default=False, type=bool, metavar='N', help='mixed precision')

parser.add_argument('--warmup', default=2, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=False, help='Version')

args = parser.parse_args()
lr = args.lr
weight_decay = args.weight_decay
dim, qk_size = args.dim, args.qk_size
m_blocks = args.m_blocks
s_blocks = args.s_blocks
dropout = args.dropout
batch_size = args.batch_size
warmup = args.warmup
width, height, channels = args.width, args.height, args.channels
patch_size = args.patch_size

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == '__main__':
    

    if args.dataset == 'mnist':
        train_dataset, val_dataset  = create_mnist_datasets(height,width)
    else:
        train_dataset, val_dataset  = create_cifar_datasets(height,width)
    num_workers = args.workers
    # Create TensorDataset and DataLoader
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    

    config = ConfigVil( n_classes= args.classes,
                        height = height,
                        width = width ,
                        patch_size= patch_size ,
                        channels = channels ,
                         m_layers = m_blocks , 
                         dim = dim , 
                         classif= args.classif_mode,
                         mlp_dim = dim*2,
                         qk_size = qk_size , 
                         dropout_rate = dropout)


    model = ViL(config)
    total_params = sum(p.numel() for p in model.parameters())
    # Print the number of parameters
    print(f"Number of parameters: {total_params}")
    model = model.to(device)
    optim_groups = model.no_weight_decay()

    optimizer = torch.optim.AdamW(
        (
            {"weight_decay": weight_decay, "params": optim_groups[1]},
            {"weight_decay": 0.0, "params": optim_groups[0]},
        ),
        lr=lr,
    )
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss().to(device)
    num_epochs = args.epochs
    min_steps = ceil(len(train_dataset)/batch_size)
    scheduler = None 
    """CosineWithLinearWarmup( optimizer, 
                                        warmup_steps=min_steps*warmup, 
                                        total_steps=int(args.epochs * min_steps),
                                        max_lr=lr,
                                        min_lr=1e-5)"""

    # Train the model
    best_loss = float('inf')
    torch.autograd.set_detect_anomaly(True)
    start_epoch = -1
    final_epoch = args.epochs
    if args.resume:
        checkpoint = torch.load('ckpt_lm.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    print(Fore.LIGHTGREEN_EX+'='*100)
    print("[INFO] Begin training for {0} epochs".format(final_epoch-start_epoch-1))
    print('='*100+Style.RESET_ALL)
    # Initialize TensorBoard writer
    writer = SummaryWriter('vision_lstm_xp')
    step = 0
    for epoch in range(start_epoch+1,final_epoch):
        train_loss,train_acc = train_epoch(model,train_loader,optimizer,scheduler,criterion,scaler,args.amp,device,writer,epoch,step)
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        if epoch%args.freq==0 or epoch==args.epochs-1:
            valid_loss,val_acc = validate(model,val_loader,criterion,device,writer,epoch)
            writer.add_scalar('Loss/Validation', valid_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {valid_loss}")
            print(f"Epoch: {epoch+1}, Train Accuarcy: {train_acc}, Val Acuracy: {val_acc}")
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        #'scheduler': scheduler.state_dict(),
                    }, f"ckpt_lm.pt")
    writer.close()
    print(Fore.GREEN+'='*100)
    print("[INFO] End training")
    print('='*100+Style.RESET_ALL)