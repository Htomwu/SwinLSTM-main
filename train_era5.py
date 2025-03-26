from utils import *
import torch.nn as nn
from configs import get_args
from functions import train, test
from torch.utils.data import DataLoader
from dataset_era5 import ERA5SPIDataset, data_provider
import numpy as np

def setup(args):
    if args.model == 'SwinLSTM-B':
        from SwinLSTM_B_d_cnn_element_fusion import SwinLSTM
        model = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths=args.depths, num_heads=args.heads_number,
                         window_size=args.window_size, drop_rate=args.drop_rate,
                         attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate).to(args.device)

    if args.model == 'SwinLSTM-D':
        from SwinLSTM_D_slstm import SwinLSTM
        model = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()


    # train_dataset =ERA5SPIDataset(args, split='train')
    # valid_dataset =ERA5SPIDataset(args, split='valid')
    #
    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
    #                           num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    #
    # valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size,
    #                          num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)
    train_loader, valid_loader = data_provider(args,
                                               dataset_name='spi',  # 设置为 'spi' 数据集
                                               train_data_paths=args.spi_data_dir,  # 传入训练数据路径
                                               valid_data_paths=args.spi_data_dir,  # 传入验证数据路径
                                               batch_size=args.train_batch_size,
                                               img_width=args.input_size[0],  # 假设是 img_size
                                               seq_length=args.num_frames_input,  # 使用 seq_length 参数
                                               injection_action=None,  # 如果有注入动作参数
                                               is_training=True)  # 是否是训练模式

    return model, criterion, optimizer, train_loader, valid_loader

def main():
    args = get_args()
    set_seed(args.seed)
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    model, criterion, optimizer, train_loader, valid_loader = setup(args)

    train_losses, valid_losses = [], []
    
    best_metric = (0, float('inf'), float('inf'))

    for epoch in range(args.epochs):

        start_time = time.time()
        train_loss = train(args, logger, epoch, model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        plot_loss(train_losses, 'train', epoch, args.res_dir, 1)

        if (epoch + 1) % args.epoch_valid == 0:

            valid_loss, mse, rmse, r2, mae, ssim, psnr = test(args, logger, epoch, model, valid_loader, criterion, cache_dir)

            valid_losses.append(valid_loss)
            
            plot_loss(valid_losses, 'valid', epoch, args.res_dir, args.epoch_valid)

            if mse < best_metric[1]:
                torch.save(model.state_dict(), f'{model_dir}/trained_model_state_dict')
                best_metric = (epoch, mse, r2, mae, ssim, psnr)

            logger.info(f'[Current Best] EP:{best_metric[0]:04d} RMSE:{best_metric[1]:.4f} R²:{best_metric[2]:.4f} MAE:{best_metric[3]:.4f} SSIM:{best_metric[4]:.4f} PSNR:{best_metric[5]:.4f}')

        print(f'Time usage per epoch: {time.time() - start_time:.0f}s')

if __name__ == '__main__':
    main()
