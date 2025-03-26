import numpy as np
import torch
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils import compute_metrics, visualize
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

scaler = amp.GradScaler()


def visualize2(inputs, targets, outputs, epoch, batch_idx, cache_dir):
    """
    可视化输入、预测和真实图像。

    Args:
        inputs (torch.Tensor): 输入图像序列，形状为 (batch_size, seq_length, channels, height, width)。
        targets (torch.Tensor): 目标图像序列，形状为 (batch_size, seq_length, channels, height, width)。
        outputs (torch.Tensor): 模型预测输出，形状为 (batch_size, seq_length, channels, height, width)。
        epoch (int): 当前训练的 epoch。
        batch_idx (int): 当前 batch 的索引。
        cache_dir (str): 图像保存的目录。
    """
    if epoch % 10 == 0:  # 每 10 个 epoch 可视化一次
        # 创建子图
        fig, axs = plt.subplots(1, 6, figsize=(25, 5))  # 共 6 张图像：前 4 张输入 + 预测 + 真实值

        # 可视化输入的前 4 张图像
        for j in range(4):
            axs[j].imshow(inputs[0, j, 0].cpu(), cmap='RdYlBu')
            axs[j].set_title(f'Input t={j}')

        # 可视化预测的第 5 张图像
        axs[4].imshow(outputs[0, 0, 0].cpu(), cmap='RdYlBu')
        axs[4].set_title('Prediction t=4')

        # 可视化真实的第 5 张图像（目标）
        axs[5].imshow(targets[0, 0, 0].cpu(), cmap='RdYlBu')
        axs[5].set_title('True t=4')

        # 保存图像
        save_path = f"{cache_dir}/epoch_{epoch:04d}_batch_{batch_idx:03d}.png"
        plt.savefig(save_path)
        plt.close(fig)


def model_forward_single_layer(model, inputs, targets_len, num_layers):
    outputs = []
    states = [None] * len(num_layers)

    inputs_len = inputs.shape[1]
    
    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states = model(inputs[:, i], states)
        outputs.append(output)

    for i in range(targets_len):
        output, states = model(last_input, states)
        outputs.append(output)
        last_input = output

    return outputs


def model_forward_multi_layer(model, inputs, targets_len, num_layers):
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        outputs.append(output)

    for i in range(targets_len):
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs


def train(args, logger, epoch, model, train_loader, criterion, optimizer):
    model.train()
    num_batches = len(train_loader)
    losses = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])
        targets_len = targets.shape[1]
        with autocast():
            if args.model == 'SwinLSTM-B':
                outputs = model_forward_single_layer(model, inputs, targets_len, args.depths)

            if args.model == 'SwinLSTM-D':
                outputs = model_forward_multi_layer(model, inputs, targets_len, args.depths_down)

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)
            loss = criterion(outputs, targets_)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        if batch_idx and batch_idx % args.log_train == 0:
            logger.info(f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f}')

    return np.mean(losses)


def test(args, logger, epoch, model, test_loader, criterion, cache_dir):
    model.eval()
    num_batches = len(test_loader)
    losses, mses, rmses, r2s, maes, ssims, psnrs = [], [], [], [], [], [], []

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])
            targets_len = targets.shape[1]

            if args.model == 'SwinLSTM-B':
                outputs = model_forward_single_layer(model, inputs, targets_len, args.depths)

            if args.model == 'SwinLSTM-D':
                outputs = model_forward_multi_layer(model, inputs, targets_len, args.depths_down)

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)

            # 计算损失
            losses.append(criterion(outputs, targets_).item())

            inputs_len = inputs.shape[1]
            outputs = outputs[:, inputs_len - 1:]

            # 计算各指标
            mse, rmse, r2, mae, avg_ssim, avg_psnr = compute_metrics(outputs, targets)
            mses.append(mse)
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)
            ssims.append(avg_ssim)
            psnrs.append(avg_psnr)

            if batch_idx and batch_idx % args.log_valid == 0:
                logger.info(
                    f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} '
                    f'Loss:{np.mean(losses):.6f} MSE:{np.mean(mses):.4f} '
                    f'RMSE:{np.mean(rmses):.4f} MAE:{np.mean(maes):.4f} '
                    f'R²:{np.mean(r2s):.4f} SSIM:{np.mean(ssims):.4f} '
                    f'PSNR:{np.mean(psnrs):.4f}'
                )

            visualize2(inputs, targets, outputs, epoch, batch_idx, cache_dir)

    return np.mean(losses), np.mean(mses), np.mean(rmses), np.mean(r2s), np.mean(maes), np.mean(ssims), np.mean(psnrs)