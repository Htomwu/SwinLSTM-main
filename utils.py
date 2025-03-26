import os
import time
import torch
import random
import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
matplotlib.use('agg')

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def visualize(inputs, targets, outputs, epoch, idx, cache_dir):
    _, axarray = plt.subplots(3, targets.shape[1], figsize=(targets.shape[1] * 5, 10))
    if targets.shape[1] == 1:
        axarray =np.expand_dims(axarray, axis=1)
    for t in range(targets.shape[1]):
        axarray[0][t].imshow(inputs[0, t, 0].detach().cpu().numpy(), cmap='gray')
        axarray[1][t].imshow(targets[0, t, 0].detach().cpu().numpy(), cmap='gray')
        axarray[2][t].imshow(outputs[0, t, 0].detach().cpu().numpy(), cmap='gray')

    plt.savefig(os.path.join(cache_dir, '{:03d}-{:03d}.png'.format(epoch, idx)))
    plt.close()

def plot_loss(loss_records, loss_type, epoch, plot_dir, step):
    plt.plot(range((epoch + 1) // step), loss_records, label=loss_type)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, '{}_loss_records.png'.format(loss_type)))
    plt.close()

def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    
def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()

# cite the 'PSNR' code from E3D-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def compute_metrics(predictions, targets):
    """
    计算预测和真实值之间的常用指标：MSE、RMSE、R²、MAE、SSIM、PSNR。

    Args:
        predictions (torch.Tensor): 预测结果，形状为 (batch_size, seq_len, channels, height, width)。
        targets (torch.Tensor): 真实值，形状为 (batch_size, seq_len, channels, height, width)。

    Returns:
        tuple: 包括 MSE、RMSE、R²、MAE、平均 SSIM 和平均 PSNR。
    """
    # 调整数据形状
    targets = targets.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    predictions = predictions.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    batch_size = predictions.shape[0]
    seq_len = predictions.shape[1]

    # 初始化指标
    total_ssim = 0
    total_psnr = 0

    # 逐样本逐帧计算 SSIM 和 PSNR
    for batch in range(batch_size):
        for frame in range(seq_len):
            true_img = targets[batch, frame].squeeze()  # 取出单帧
            pred_img = predictions[batch, frame].squeeze()  # 取出单帧

            # 确保形状一致
            assert true_img.shape == pred_img.shape, f"Shape mismatch: {true_img.shape} vs {pred_img.shape}"

            # 计算 SSIM 和 PSNR
            total_ssim += ssim(true_img, pred_img, data_range=pred_img.max() - pred_img.min())
            total_psnr += psnr(true_img, pred_img, data_range=pred_img.max() - pred_img.min())

    # 计算平均 SSIM 和 PSNR
    avg_ssim = total_ssim / (batch_size * seq_len)
    avg_psnr = total_psnr / (batch_size * seq_len)

    # 计算全局指标
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())

    return mse, rmse, r2, mae, avg_ssim, avg_psnr

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(args):

    cache_dir = os.path.join(args.res_dir, 'cache')
    check_dir(cache_dir)

    model_dir = os.path.join(args.res_dir, 'model')
    check_dir(model_dir)

    log_dir = os.path.join(args.res_dir, 'log')
    check_dir(log_dir)

    return cache_dir, model_dir, log_dir

def init_logger(log_dir):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + '.log'),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging
