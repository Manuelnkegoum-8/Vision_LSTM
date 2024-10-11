import torch
import torch.nn as nn
import logging
from .helpers import calculate_accuracy
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler,mixed_prec, device,writer,epoch,step):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.0
    model.train()

    # Wrap dataloader with tqdm for a progress bar
    with tqdm(total=len(dataloader), desc="Training", colour='blue', ncols=100) as pbar:
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            images, labels = data
            bs = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            with autocast(enabled=mixed_prec):
                preds = model(images)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            writer.add_scalar('lr/Train', optimizer.param_groups[0]['lr'], step)
            avg_loss += loss.item() * bs
            acc = calculate_accuracy(preds,labels)
            avg_acc += acc * bs
            n += bs
            # Update the progress bar with the current loss
            pbar.set_postfix({'loss':loss.item(), 'accuracy': acc})
            pbar.update(1)
            step+=1
    avg_loss /= n
    avg_acc /= n
    return avg_loss,avg_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device,writer,epoch):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.0
    model.eval()

    # Wrap dataloader with tqdm for a progress bar
    with tqdm(total=len(dataloader), desc="Validation", colour='yellow', ncols=100) as pbar:
        for i, data in enumerate(dataloader):
            images, labels = data
            bs = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            avg_loss += loss.item() * bs
            acc = calculate_accuracy(preds,labels)
            avg_acc += acc * bs
            n += bs

            # Update the progress bar with the current loss
            pbar.set_postfix({'loss':loss.item(), 'accuracy': acc})
            pbar.update(1)

    avg_loss /= n
    avg_acc /= n
    return avg_loss,avg_acc

        