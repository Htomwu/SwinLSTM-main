from utils import *
import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.ndimage import zoom
# 修改后的 data_provider
def data_provider(args, dataset_name, train_data_paths, valid_data_paths, batch_size, img_width, seq_length, injection_action, is_training=True):
    # 检查数据集名称
    if dataset_name != 'spi':
        raise ValueError(f"Dataset {dataset_name} not supported!")

    # 加载训练和验证数据集路径
    train_data_list = train_data_paths
    valid_data_list = valid_data_paths

    # 使用 xarray 读取 .nc 文件
    ds_train = xr.open_dataset(train_data_list)  # 假设这里只有一个训练文件，若多个文件可以扩展
    data = ds_train['spi'].values

    # 数据预处理：去除 NaN 数据，并填充
    data = data[~np.isnan(data).all(axis=(1, 2))]
    data = np.where(np.isnan(data), np.nanmean(data, axis=(0, 1), keepdims = True), data)

    # 步骤 3: 插值 - 将数据从 (360, 720) 缩小到 (180, 360)
    time_steps, lat, lon = data.shape

    # 设置目标分辨率
    new_lat = 180
    new_lon = 360

    # 创建原始的经纬度网格
    latitudes = np.linspace(-90, 90, lat)
    longitudes = np.linspace(-180, 180, lon)

    # 创建新的目标经纬度网格
    new_latitudes = np.linspace(-90, 90, new_lat)
    new_longitudes = np.linspace(-180, 180, new_lon)

    # 新数组用于存储调整后的数据
    resized_data = np.zeros((time_steps, new_lat, new_lon))  # 新数据数组形状为 (time_steps, 180, 360)

    # 使用 zoom 或插值方法将每个时间步的数据调整到新分辨率
    for t in range(time_steps):
        # 获取当前时间步的二维数据
        current_data = data[t, :, :]

        # 进行下采样
        interpolated_data = zoom(current_data, (new_lat / lat, new_lon / lon), order=3)

        # 确保插值后的数据形状为 (180, 360)
        assert interpolated_data.shape == (new_lat, new_lon), f"插值后的数据形状不匹配: {interpolated_data.shape}"

        # 将插值后的数据存入新的数组
        resized_data[t] = interpolated_data  # 直接替换当前时间步的数据
    # 数据标准化处理
    scaler = MinMaxScaler()
    time_steps, lat, lon = resized_data.shape
    resized_data = scaler.fit_transform(resized_data.reshape(time_steps, -1)).reshape(time_steps, lat, lon)

    # 划分数据集：80% 训练集，20% 测试集
    total_sequences = len(resized_data) - seq_length + 1
    indices = list(range(total_sequences))
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # 创建训练和测试数据集
    train_dataset = ERA5SPIDataset(args, split='train', data=resized_data, indices=train_indices, seq_length=seq_length, batch_size=batch_size)
    valid_dataset = ERA5SPIDataset(args, split='valid', data=resized_data, indices=test_indices, seq_length=seq_length, batch_size=batch_size)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    return train_loader, valid_loader

# 修改后的 ERA5SPIDataset
class ERA5SPIDataset(Dataset):
    def __init__(self, args, split, data, indices, seq_length, batch_size):
        self.data = data
        self.indices = indices
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.split = split
        self.num_frames_total = seq_length + 1  # 输入序列长度

    def __getitem__(self, index):
        # 获取输入数据序列
        start_idx = index
        end_idx = start_idx + self.seq_length + 1

        # 获取 SPI 数据
        spi_data = self.data[start_idx:end_idx]

        # 将数据转为张量并调整维度 (time, 1, lat, lon)
        data = torch.from_numpy(spi_data[..., np.newaxis]).permute(0, 3, 1, 2).contiguous()

        # 分割输入和目标
        inputs = data[:self.seq_length]
        targets = data[self.seq_length:]

        return inputs, targets

    def __len__(self):
        return len(self.indices)  # 返回数据集大小

