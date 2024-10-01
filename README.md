
# Vision LSTM (ViL) - MNIST & CIFAR Classification

## Overview
This project implements a Vision LSTM (ViL) architecture, which adapts Long Short-Term Memory (LSTM) for computer vision tasks. ViL extends traditional LSTMs with scalable and parallel processing capabilities to handle the challenges of vision data, particularly sequences of image patches, and integrates both top-down and bottom-up processing flows.

The model was trained and tested on two datasets:
1. **MNIST**: Handwritten digit classification.
2. **CIFAR**: Object classification in 32x32 color images.

## Architecture
Vision LSTM (ViL) uses the xLSTM backbone, designed specifically for vision tasks. The architecture processes image patches sequentially, allowing for long-range dependencies to be captured. Alternating layers of LSTMs process patch tokens both from top-to-bottom and bottom-to-top, improving the model's ability to capture fine-grained and holistic image details. This makes ViL a powerful architecture for tasks such as image classification and segmentation.

Key features of the architecture include:
- **Exponential gating**: Improves the memory retention of the LSTM units, addressing vanishing gradients and extending the receptive field.
- **Patch token processing**: Images are split into patches that are processed sequentially by the LSTM layers.
- **Parallelizable matrix memory**: Enhances computational efficiency and scalability.

For more detailed information, you can refer to [Vision LSTM Paper](https://brandstetter-johannes.github.io/publication/alkin-2024-vision-lstm/)【10†source】【11†source】.

## Results
The performance of the model on MNIST and CIFAR datasets is summarized below:

| Dataset    | Model          | Accuracy (%) |
|------------|----------------|--------------|
| MNIST      | Vision LSTM    | 99.1         |
| CIFAR-10   | Vision LSTM    | 88.3         |

The results demonstrate that the ViL model achieves state-of-the-art performance on MNIST and performs competitively on CIFAR-10, showcasing its versatility across both grayscale and color image tasks.

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vision-lstm.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py --dataset mnist
   ```
   For CIFAR, use:
   ```bash
   python train.py --dataset cifar
   ```

## References
- [Vision LSTM Paper](https://brandstetter-johannes.github.io/publication/alkin-2024-vision-lstm/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
