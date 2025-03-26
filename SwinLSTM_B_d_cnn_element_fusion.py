import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


class DeformableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable_groups=1):
        super(DeformableConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv layers for offsets and mask
        self.offset_conv = nn.Conv1d(in_channels, deformable_groups * kernel_size,
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self.mask_conv = nn.Conv1d(in_channels, deformable_groups * kernel_size,
                                   kernel_size=kernel_size, stride=stride, padding=padding)

        # Regular convolution layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation)

    def forward(self, x):
        B, L, C = x.size()  # B: batch size, L: length (time steps), C: channels

        # Fix the input shape for Conv1D
        x = x.permute(0, 2, 1)  # Change shape from (B, L, C) to (B, C, L)

        # Generate offsets and masks (these will be (B, deformable_groups * 2 * kernel_size, L))
        offset = self.offset_conv(x)  # Shape: (B, deformable_groups * 2 * kernel_size, L)
        mask = torch.sigmoid(self.mask_conv(x))  # Shape: (B, deformable_groups * kernel_size, L)

        # Deformable Convolution logic
        # Offset reshaping
        offset = offset.view(B, self.kernel_size, L)  # For 1D: each location gets 2 offsets (horizontal shift)
        mask = mask.view(B, self.kernel_size, L)  # Each location gets a mask

        # Apply deformable convolution manually: sampling based on the offsets
        output = self.deform_conv1d(x, offset, mask)

        # Reshape output back to (B, L, C) format
        output = output.view(B, L, self.out_channels)
        return output

    def deform_conv1d(self, x, offset, mask):
        """
        Custom implementation for deformable convolution 1D
        x: input tensor of shape (B, C, L)
        offset: calculated offsets for deformable convolution
        mask: calculated mask for deformable convolution
        """
        B, C, L = x.size()

        # Initialize the output tensor
        output = torch.zeros((B, L, self.out_channels), device=x.device)

        # Apply the deformable convolution (the key part is here)
        # Instead of looping, we'll use broadcasting to compute everything in parallel.

        # Expand x to (B, C, kernel_size, L)
        x_expanded = x.unsqueeze(2).expand(B, C, self.kernel_size, L)  # (B, C, kernel_size, L)

        # Correcting offset reshaping
        # offset is of shape (B, 2 * kernel_size, L)
        offset_expanded = offset.unsqueeze(1).expand(B, C, self.kernel_size, L)  # (B, C, kernel_size, L)

        # Ensure offset expansion is consistent with C and kernel_size
        # We want the offset values to modify the indices accordingly.

        # Here, you need to apply the actual sampling logic for deformable convolution
        sampled_values = self.sample_from_offset(x_expanded, offset_expanded)

        # Apply mask
        mask_expanded = mask.unsqueeze(1).expand(B, C, self.kernel_size, L)  # Expand mask to (B, C, kernel_size, L)

        # Apply mask and sum over kernel size dimension
        output = (sampled_values * mask_expanded).sum(dim=2)  # Sum over kernel size dimension

        return output

    def sample_from_offset(self, x, offset):
        """
        基于 offset 对 x 做 1D 线性插值采样.
        x: (B, C, K, L)
        offset: (B, C, K, L) -- 每个位置对应一个偏移值(可为小数)
        返回: (B, C, K, L) -- 在新位置采样到的特征.
        """

        B, C, K, L = x.shape

        # 1) 生成原始索引网格 i: 大小 (1, 1, 1, L), 扩展到 (B, C, K, L)
        #    每个 "l" 表示原序列中的整数位置.
        device = x.device
        i = torch.arange(L, device=device).view(1, 1, 1, L).float()  # shape (1,1,1,L)
        i = i.expand(B, C, K, L)  # (B,C,K,L)

        # 2) 计算新的采样位置: pos = i + offset
        #    offset[b,c,k,l] 可能是正也可能是负, 带小数.
        pos = i + offset  # (B,C,K,L)

        # 3) 做边界裁剪: 确保采样点在 [0, L-1] 范围内
        pos = pos.clamp(0, L - 1)

        # 4) 找到 floor_pos 和 ceil_pos, 以及插值系数 alpha
        floor_pos = torch.floor(pos).long()  # 向下取整, (B,C,K,L), int 索引
        ceil_pos = torch.clamp(floor_pos + 1, max=L - 1)  # (B,C,K,L)
        alpha = pos - floor_pos.float()  # (B,C,K,L), 范围 [0,1)

        # 5) 利用 gather 在 dim=3(序列维度) 上取 floor_pos / ceil_pos 对应的特征
        #    x[..., idx] 变成 x.gather(dim=3, index=idx), 其中 idx 形状与 x 相同
        x_floor = x.gather(dim=3, index=floor_pos)  # (B,C,K)的展开 + (B,C,K,L)索引 → (B,C,K,L)
        x_ceil = x.gather(dim=3, index=ceil_pos)  # 同上

        # 6) 做线性插值: val = x_floor*(1 - alpha) + x_ceil*alpha
        #    alpha.shape = x_floor.shape = (B,C,K,L)
        sampled_values = x_floor * (1 - alpha) + x_ceil * alpha

        return sampled_values
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LocalGlobalFusionLayer(nn.Module):
    def __init__(self, local_dim, global_dim, fusion_dim):
        super(LocalGlobalFusionLayer, self).__init__()
        self.local_fc = nn.Linear(local_dim, fusion_dim)
        self.global_fc = nn.Linear(global_dim, fusion_dim)
        self.fusion_fc = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, local_features, global_features):
        local_out = self.local_fc(local_features)
        global_out = self.global_fc(global_features)
        # 逐元素相乘
        fusion_out = local_out * global_out
        fusion_out = F.relu(fusion_out)
        return self.fusion_fc(fusion_out)


class SwinLSTMCellWithFusion(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, fusion_dim=128):
        super(SwinLSTMCellWithFusion, self).__init__()

        # 使用 Deformable CNN 提取局部特征
        self.deformable_conv = DeformableConv1d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

        # Swin Transformer 提取全局特征
        self.Swin = SwinTransformerBlocks(dim=dim, input_resolution=input_resolution, depth=depth,
                                          num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path, norm_layer=norm_layer)

        # 融合层
        self.fusion_layer = LocalGlobalFusionLayer(local_dim=dim, global_dim=dim, fusion_dim=fusion_dim)

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)
        else:
            hx, cx = hidden_states

        # 应用 Deformable CNN 提取局部特征
        local_features = self.deformable_conv(xt)

        # 获取 Swin Transformer 提取的全局特征
        global_features = self.Swin(xt, hx)

        # 融合局部和全局特征
        fused_features = self.fusion_layer(local_features, global_features)

        # fused_features = local_features

        # LSTM 门计算
        gate = torch.sigmoid(fused_features)
        cell = torch.tanh(fused_features)

        # 更新细胞状态和隐藏状态
        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        hx = hy
        cx = cy

        return hx, (hx, cx)


class STconvertWithFusion(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, fusion_dim=128):
        super(STconvertWithFusion, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution

        self.PatchInflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = SwinLSTMCellWithFusion(dim=embed_dim,
                                           input_resolution=(patches_resolution[0], patches_resolution[1]),
                                           depth=depths[i_layer],
                                           num_heads=num_heads[i_layer],
                                           window_size=window_size,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop_rate, attn_drop=attn_drop_rate,
                                           drop_path=drop_path_rate,
                                           norm_layer=norm_layer,
                                           fusion_dim=fusion_dim)  # 传递 fusion_dim

            self.layers.append(layer)

    def forward(self, x, h):
        x = self.patch_embed(x)

        hidden_states = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, h[index])
            hidden_states.append(hidden_state)

        x = torch.sigmoid(self.PatchInflated(x))

        return hidden_states, x
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.
        patch_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.ConvT = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                        stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.ConvT(x)

        return x
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=2, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.red = nn.Linear(2 * dim, dim)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlocks(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(SwinTransformerBlocks, self).__init__()
        self.layers = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, xt, hx):

        outputs = []

        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)

            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)

                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)

        return outputs[-1]



class SwinLSTM(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths,
                 num_heads, window_size, drop_rate, attn_drop_rate, drop_path_rate, fusion_dim=128):
        super(SwinLSTM, self).__init__()

        self.ST = STconvertWithFusion(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim, depths=depths,
                                      num_heads=num_heads, window_size=window_size, drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      fusion_dim=fusion_dim)

    def forward(self, input, states):
        states_next, output = self.ST(input, states)
        return output, states_next