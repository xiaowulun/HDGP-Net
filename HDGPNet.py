import typing as t
import torch
import torch.nn.functional as F
from torch import nn

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 第一个卷积
            nn.BatchNorm2d(mid_channels),  # 批归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  # 第二个卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )



class DropBlock2D(nn.Module):
    r"""Randomly drops continuous blocks of feature map.
    Standard implementation with correct Gamma calculation.
    """

    def __init__(self, drop_prob=0.1, block_size=5):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # 1. 训练模式且概率>0才执行
        if not self.training or self.drop_prob == 0.0:
            return x

        # 2. 计算正确的 gamma (采样概率)
        gamma = self._compute_gamma(x)

        # 3. 生成 mask (N, 1, H, W)
        # 注意：这里mask是 (N, 1, ...) 表示所有通道共享同一个 mask (Sync Drop)
        # 这对于模拟"结构性遮挡"至关重要
        mask = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) < gamma).float()

        # 4. 使用 MaxPool 将种子点膨胀成 block
        # padding 保证 mask 大小不变
        block_mask = self._compute_block_mask(mask)

        # 5. 归一化 (Normalization)
        # 统计实际被保留的像素比例，进行数值缩放，保证期望值不变
        count = block_mask.size()[0] * block_mask.size()[2] * block_mask.size()[3]
        count_ones = block_mask.sum()

        # 归一化操作：x * mask * (总像素 / 保留像素)
        return x * block_mask * (count / (count_ones + 1e-6))

    def _compute_block_mask(self, mask):
        # 形态学膨胀：将 seed 点变成 block
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        # 如果 block_size 是偶数，padding 会导致多出一行/列，需要裁剪
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        # 反转：1代表保留，0代表扔掉
        block_mask = 1 - block_mask
        return block_mask

    def _compute_gamma(self, x):
        # [核心修正] 根据 Feature Map 大小动态计算 Gamma
        _, _, H, W = x.shape

        # 确保 block_size 不超过特征图大小
        clipped_block_size = min(self.block_size, min(H, W))

        # 论文标准公式
        # 有效采样区域面积
        valid_area = (H - clipped_block_size + 1) * (W - clipped_block_size + 1)
        total_area = H * W

        gamma = (self.drop_prob * total_area) / (clipped_block_size ** 2 * valid_area)
        return gamma

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + residual)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class StripBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.pre_conv = BasicConv(in_channels, out_channels, 1)

        padding = kernel_size // 2
        # 条带卷积在 out_channels 上进行
        self.conv_h = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding),
                                groups=out_channels, bias=False)
        self.conv_w = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0),
                                groups=out_channels, bias=False)

        # self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)  # 在相加后进行BN
        self.channel_attn = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.pre_conv(x)
        x_h = self.conv_h(x)
        x_w = self.conv_w(x)
        return self.channel_attn(self.bn(x_h + x_w))

class LFR(nn.Module):

    def __init__(self, in_channels, out_channels, n=2):
        super().__init__()
        self.n = n
        self.conv_in = BasicConv(in_channels, out_channels, 1)

        hidden_channels = out_channels // 2
        self.bottlenecks = nn.ModuleList(BasicBlock(hidden_channels, hidden_channels) for _ in range(n))

        self.conv_out = BasicConv(hidden_channels * (1 + n), out_channels, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x1, x2 = x.chunk(2, 1)
        # 存储所有中间输出
        y = [x1]
        for bottleneck in self.bottlenecks:
            x2 = bottleneck(x2)
            y.append(x2)

        return self.conv_out(torch.cat(y, 1))


# Directional-Aware Aggregation module(DAA)
class DAA(nn.Module):

    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super(DAA, self).__init__()

        # ... Path 1 & Path 2 definitions (保持不变) ...
        self.path1_reuse = LFR(in_channels, out_channels, n=2)
        self.path2_global = StripBlock(in_channels, out_channels, kernel_size=7)

        # --- 最终融合层 ---
        self.fusion_conv = ConvBN(out_channels * 2, out_channels, 1)

        # 残差投影 (保持不变)
        if in_channels != out_channels:
            self.residual_proj = BasicConv(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()

        self.drop_block = DropBlock2D(drop_prob=drop_prob, block_size=3)

        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. 计算残差边
        residual = self.residual_proj(x)

        # 2. 两路特征提取
        out1 = self.path1_reuse(x)
        out2 = self.path2_global(x)

        # 3. 拼接
        concatenated = torch.cat([out1, out2], dim=1)

        # 4. 融合
        fused = self.fusion_conv(concatenated)
        fused = self.drop_block(fused)

        output = self.final_activation(fused + residual)

        return output


# 层级交叉门控融合 (Hierarchical Cross Gated Fusion, HCGF)
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_compress = torch.cat([avg_out, max_out], dim=1)

        return self.sigmoid(self.conv(x_compress))

class HCGF(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 1. Context Purifier (保持不变)
        self.context_purifier = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.main_sa = SpatialAttention(kernel_size=7)
        self.context_sa = SpatialAttention(kernel_size=7)

        self.fusion_gate_generator = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, main_feature, context_feature):
        # 1. Purify
        purified_context = self.context_purifier(context_feature)

        # 生成基于方差的空间 Mask
        map_main = self.main_sa(main_feature)
        map_context = self.context_sa(purified_context)

        # 双向校准：让特征图聚焦在"高频边缘"区域
        refined_main = main_feature * map_context
        refined_context = purified_context * map_main

        # 3. Adaptive Gated Fusion
        gate_input = torch.cat([refined_main, refined_context], dim=1)
        fusion_gate = self.fusion_gate_generator(gate_input)

        fused_feature = fusion_gate * refined_main + (1 - fusion_gate) * refined_context

        return fused_feature


class AggregativeFusionModule(nn.Module):

    def __init__(self, channels, num_support_branches):
        super().__init__()
        #  辅路聚合器
        support_channels = num_support_branches * channels
        self.context_aggregator = nn.Sequential(
            nn.Conv2d(support_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.hcgf_unit = HCGF(channels)

    def forward(self, feature_maps: t.List[torch.Tensor]):
        main_feature = feature_maps[0]
        support_features = feature_maps[1:]

        concatenated_support = torch.cat(support_features, dim=1)

        context_feature = self.context_aggregator(concatenated_support)
        fused_feature = self.hcgf_unit(main_feature, context_feature)

        return fused_feature

# --- 一个轻量级的融合模块 ---
class FusionBlock(nn.Module):

    def __init__(self, high_res_channels, low_res_channels):
        super(FusionBlock, self).__init__()
        # 1x1 卷积用于将上采样后的低分辨率特征的通道数对齐到高分辨率特征
        self.align_conv = nn.Conv2d(low_res_channels, high_res_channels, kernel_size=1, bias=False)
        self.bn_align = nn.BatchNorm2d(high_res_channels)

        # 3x3 卷积用于平滑和提炼融合后的特征
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(high_res_channels),
            nn.ReLU(inplace=True),
            # DropBlock2D(drop_prob=0.1, block_size=3)
        )

    def forward(self, high_res_feature, low_res_feature):
        # 1. 上采样低分辨率特征
        low_res_upsampled = F.interpolate(low_res_feature, size=high_res_feature.shape[2:], mode='bilinear',
                                          align_corners=True)
        # 2. 对齐通道
        low_res_aligned = self.bn_align(self.align_conv(low_res_upsampled))

        # 3. 将两者相加
        fused = high_res_feature + low_res_aligned

        # 4. 通过卷积进行平滑提炼
        output = self.fusion_conv(fused)
        return output


class PyramidFusionHead(nn.Module):

    def __init__(self, all_channels_list, num_classes=1, drop_prob=0.1):
        super(PyramidFusionHead, self).__init__()

        # 融合模块1: low-res (128) -> mid-res (64)
        self.fuse1 = FusionBlock(high_res_channels=all_channels_list[1],
                                 low_res_channels=all_channels_list[2])

        # 融合模块2: mid-res (64) -> high-res (32)
        self.fuse2 = FusionBlock(high_res_channels=all_channels_list[0],
                                 low_res_channels=all_channels_list[1])

        # 最终的分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(all_channels_list[0], all_channels_list[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(all_channels_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(all_channels_list[0], num_classes, kernel_size=1)
        )

        # 辅助头 (用于深度监督)
        self.aux_head = nn.Sequential(
            nn.Conv2d(all_channels_list[0], all_channels_list[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(all_channels_list[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(all_channels_list[0] // 2, num_classes, kernel_size=1)
        )

    def forward(self, all_features: t.List[torch.Tensor]):
        x_res1, x_res2, x_res3 = all_features

        # 1. 最低分辨率 (x_res3) 融合到中间分辨率 (x_res2)
        fused_res2 = self.fuse1(high_res_feature=x_res2, low_res_feature=x_res3)

        # 2. 融合后的中间分辨率 (fused_res2) 融合到最高分辨率 (x_res1)
        fused_res1 = self.fuse2(high_res_feature=x_res1, low_res_feature=fused_res2)

        # 3. 在最终的、信息最丰富的特征图上进行分类
        output = self.classifier(fused_res1)

        if self.training:
            # 辅助输出，直接从原始的、未融合的最高分辨率特征生成
            aux_output = self.aux_head(x_res1)
            return [output, aux_output]
        else:
            return output


class DirectHead(nn.Module):
    """
    [Ablation Baseline] 无 PFH 的情况
    直接把最高分辨率的特征图 (x_res1) 投影输出
    """

    def __init__(self, in_channels, num_classes=1):
        super(DirectHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, all_features: t.List[torch.Tensor]):
        # 只取最高分辨率特征 x_res1 (list的第一个)
        x_high = all_features[0]
        return self.classifier(x_high)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, num_blocks, num_channels, use_daa, use_hcgf, drop_prob, **kwargs):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.use_hcgf = use_hcgf

        # [开关 1: DAA vs Basic]
        if use_daa:
            BlockClass = DAA
            block_kwargs = {'drop_prob': drop_prob}
        else:
            BlockClass = DoubleConv  # 回退到普通卷积块
            block_kwargs = {}

        self.blocks = nn.ModuleList([
            nn.Sequential(*[BlockClass(num_channels[i], num_channels[i], **block_kwargs) for _ in range(num_blocks)])
            for i in range(num_branches)
        ])

        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels)

        if self.use_hcgf:
            self.fusion_modules = nn.ModuleList()
            for i in range(num_branches):
                module = AggregativeFusionModule(
                    channels=num_channels[i],
                    num_support_branches=num_branches - 1
                )
                self.fusion_modules.append(module)
        else:
            self.fusion_modules = None  # None 表示使用 sum 融合

        self.relu = nn.ReLU(inplace=True)  # 保持 ReLU

    def _make_fuse_layers(self, num_branches, num_channels):
        if num_branches == 1: return None
        fuse_layers = []
        for i in range(num_branches):
            fuse_branch_layers = []
            for j in range(num_branches):
                if j > i:
                    fuse_branch_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='bilinear', align_corners=True)))
                elif j < i:
                    conv3x3s = []
                    for k in range(i - j):
                        in_ch = num_channels[j] if k == 0 else num_channels[i]
                        out_ch = num_channels[i]
                        conv3x3s.append(nn.Sequential(
                            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True)))
                    fuse_branch_layers.append(nn.Sequential(*conv3x3s))
                else:
                    fuse_branch_layers.append(nn.Identity())
            fuse_layers.append(nn.ModuleList(fuse_branch_layers))
        return nn.ModuleList(fuse_layers)

    def forward(self, x: t.List[torch.Tensor]):
        for i in range(self.num_branches):
            x[i] = self.blocks[i](x[i])

        if self.num_branches == 1: return x

        fuse_branches = []
        for i in range(self.num_branches):
            branch_i_inputs = []
            for j in range(self.num_branches):
                branch_i_inputs.append(self.fuse_layers[i][j](x[j]))
            fuse_branches.append(branch_i_inputs)

        x_fused = []
        x_pre_fuse = list(x)
        for i in range(self.num_branches):
            if self.use_hcgf:
                y_transformed = self.fusion_modules[i](fuse_branches[i])
                y = self.relu(x_pre_fuse[i] + y_transformed)
            else:

                y = torch.sum(torch.stack(fuse_branches[i]), dim=0)
                y = self.relu(y)
            x_fused.append(y)

        return x_fused


class HDGFNet(nn.Module):
    def __init__(self, num_classes=1, drop_prob=0.2,
                 use_daa=True, use_hcgf=True, use_pfh=True):  # 这里直接用开关

        super(HDGFNet, self).__init__()

        # Stage 1 (Stem) 始终不变
        self.stage1 = nn.Sequential(BasicBlock(1, 32))
        self.transition1 = self._make_transition_layer([32], [32, 64])

        # Stage 2
        self.stage2 = HighResolutionModule(
            num_branches=2, num_blocks=1, num_channels=[32, 64],
            use_daa=use_daa,  # [开关传入]
            use_hcgf=use_hcgf,  # [开关传入]
            drop_prob=drop_prob,
            block=None, fuse_method=None  # 这两个参数废弃了，由开关控制
        )
        self.transition2 = self._make_transition_layer([32, 64], [32, 64, 128])

        # Stage 3
        self.stage3 = HighResolutionModule(
            num_branches=3, num_blocks=1, num_channels=[32, 64, 128],
            use_daa=use_daa,  # [开关传入]
            use_hcgf=use_hcgf,  # [开关传入]
            drop_prob=drop_prob,
            block=None, fuse_method=None
        )

        # [开关 3: Head Selection]
        if use_pfh:
            self.segmentation_head = PyramidFusionHead(
                all_channels_list=[32, 64, 128], num_classes=num_classes, drop_prob=drop_prob
            )
        else:
            # Ablation: 简单的直接输出头
            self.segmentation_head = DirectHead(
                in_channels=32, num_classes=num_classes
            )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        transition_layers = []
        for i in range(len(num_channels_cur_layer)):
            if i < len(num_channels_pre_layer):
                if num_channels_pre_layer[i] != num_channels_cur_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                transition_layers.append(nn.Sequential(
                    nn.Conv2d(num_channels_pre_layer[-1], num_channels_cur_layer[i], 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
        return nn.ModuleList(transition_layers)

    def forward(self, x):
        x = self.stage1(x)
        x_pre = x
        x_list = [self.transition1[0](x_pre) if self.transition1[0] else x_pre, self.transition1[1](x_pre)]
        x_list = self.stage2(x_list)
        x_pre = x_list
        x_list = [self.transition2[0](x_pre[0]) if self.transition2[0] else x_pre[0],
                  self.transition2[1](x_pre[1]) if self.transition2[1] else x_pre[1], self.transition2[2](x_pre[-1])]
        x_list_s3 = self.stage3(x_list)

        out = self.segmentation_head(x_list_s3)
        return out
