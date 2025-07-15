import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from einops import rearrange
from model.MobileNetV2 import mobilenet_v2


class BasicConv2d(nn.Module):
    """Basic convolution block with Conv2d + BatchNorm + ReLU"""
    
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction(nn.Module):
    """Channel reduction module"""
    
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, head=8, d_model=32, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None
        self.input_projection = nn.Linear(32, d_model)

    def self_attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        self_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            self_attn = self.dropout(self_attn)
        return torch.matmul(self_attn, value), self_attn

    def forward(self, query, key, value, mask=None):
        # Flatten spatial dimensions and permute
        query = query.flatten(start_dim=2).permute(0, 2, 1)
        key = key.flatten(start_dim=2).permute(0, 2, 1)
        value = value.flatten(start_dim=2).permute(0, 2, 1)
        
        # Project inputs
        query = self.input_projection(query)
        key = self.input_projection(key)
        value = self.input_projection(value)
        
        # Apply self-attention
        x, self.attn = self.self_attention(query, key, value, mask=mask)
        
        # Reshape output
        x = x.permute(0, 2, 1)
        embedding_dim = x.size(-1)
        d_k = h = int(embedding_dim ** 0.5)
        x = einops.rearrange(x, 'b n (d_k h) -> b n d_k h', d_k=d_k, h=h)
        
        return x


class Upsample(nn.Module):
    """Upsampling module"""
    
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target):
        x = F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True)
        return x


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism"""
    
    def __init__(self, channel):
        super(MultiScaleAttention, self).__init__()
        self.attention_branches = nn.ModuleList([SpatialAttention() for _ in range(5)])
        self.upsample = Upsample()
        self.conv_reduce = nn.Conv2d(channel * 6, channel, kernel_size=1)

    def forward(self, x0, x1, x2, x3, x4, x5):
        # Apply spatial attention to each branch
        x0_att = self.attention_branches[0](x0) * x0
        x1_att = self.attention_branches[0](x1) * x1
        x2_att = self.attention_branches[0](x2) * x2
        x3_att = self.attention_branches[0](x3) * x3
        x4_att = self.attention_branches[0](x4) * x4
        x5_att = self.attention_branches[0](x5) * x5

        # Upsample all features to x0's size
        x1_att_up = self.upsample(x1_att, x0)
        x2_att_up = self.upsample(x2_att, x0)
        x3_att_up = self.upsample(x3_att, x0)
        x4_att_up = self.upsample(x4_att, x0)
        x5_att_up = self.upsample(x5_att, x0)

        # Concatenate and reduce channels
        x_cat = torch.cat((x0_att, x1_att_up, x2_att_up, x3_att_up, x4_att_up, x5_att_up), dim=1)
        x_out = self.conv_reduce(x_cat)

        return x_out


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(x_concat)
        return self.sigmoid(attention)


class DilatedConvBlock(nn.Module):
    """Dilated convolution block for multi-scale feature extraction"""
    
    def __init__(self, in_channel, out_channel):
        super(DilatedConvBlock, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
        self.branch5 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 11), padding=(0, 5)),
            BasicConv2d(out_channel, out_channel, kernel_size=(11, 1), padding=(5, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=11, dilation=11)
        )
        
        self.multi_scale_attention = MultiScaleAttention(out_channel)
        self.conv_combine = BasicConv2d(out_channel, in_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)

        x_att = self.multi_scale_attention(x0, x1, x2, x3, x4, x5)
        x_combined = self.conv_combine(x_att)

        return x_combined + x


class MixedModule(nn.Module):
    """Mixed module combining dilated convolutions and attention mechanisms"""
    
    def __init__(self, channel):
        super(MixedModule, self).__init__()
        self.dilated_block = DilatedConvBlock(channel, channel)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channel)

    def forward(self, x):
        x_mixed = self.dilated_block(x)
        x_mixed = x_mixed * self.channel_attention(x_mixed) + x_mixed
        x_mixed = x_mixed * self.spatial_attention(x_mixed) + x_mixed
        return x_mixed + x


class ProgressiveModule(nn.Module):
    """Progressive module for multi-scale feature enhancement"""
    
    def __init__(self, channel):
        super(ProgressiveModule, self).__init__()
        self.atrous_conv1 = BasicConv2d(channel, channel, 3, padding=3, dilation=3)
        self.atrous_conv2 = BasicConv2d(channel, channel, 3, padding=5, dilation=5)
        self.atrous_conv3 = BasicConv2d(channel, channel, 3, padding=7, dilation=7)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0))
        )

        self.conv_cat1 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_cat2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_cat3 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_reduce = BasicConv2d(channel, channel, 1)

        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channel)
        
        self.sal_conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x_atr1 = self.atrous_conv1(x)
        s_mfeb1 = self.conv_cat1(torch.cat((x1, x_atr1), 1)) + x
        
        x2 = self.branch2(s_mfeb1)
        x_atr2 = self.atrous_conv2(s_mfeb1)
        s_mfeb2 = self.conv_cat2(torch.cat((x2, x_atr2), 1)) + s_mfeb1 + x
        
        x3 = self.branch3(s_mfeb2)
        x_atr3 = self.atrous_conv3(s_mfeb2)
        s_mfeb3 = self.conv_cat3(torch.cat((x3, x_atr3), 1)) + s_mfeb1 + s_mfeb2 + x
        
        x_reduced = self.conv_reduce(s_mfeb3)
        x_ca = self.channel_attention(x_reduced) * x_reduced
        x_mixed = self.sal_conv(self.spatial_attention(x_ca) * x_ca) + s_mfeb1 + s_mfeb2 + s_mfeb3 + x

        return x_mixed


class TransBasicConv2d(nn.Module):
    """Transpose convolution block"""
    
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FeatureExtractor(nn.Module):
    """Feature extraction module"""
    
    def __init__(self, channel):
        super(FeatureExtractor, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 1)
        self.conv2 = BasicConv2d(channel, channel, 1)
        self.conv3 = BasicConv2d(channel, channel, 1)
        self.conv4 = BasicConv2d(channel, channel, 1)
        self.conv5 = BasicConv2d(channel, channel, 1)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)
        return x1, x2, x3, x4, x5


class ConvUpsample(nn.Module):
    """Convolution with upsampling"""
    
    def __init__(self, channel):
        super(ConvUpsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class AdaptivePoolingModule(nn.Module):
    """Adaptive pooling module"""
    
    def __init__(self, stride=2):
        super(AdaptivePoolingModule, self).__init__()
        self.stride = stride
        self.avg_pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        self.max_pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x1, x2):
        avg_pool = self.avg_pool(x1)
        max_pool = self.max_pool(x2)
        result = torch.norm(abs(avg_pool - max_pool), p=2, dim=1, keepdim=True)
        return result


class ASAM(nn.Module):
    """Adaptive Spatial Attention Module"""
    
    def __init__(self, channel):
        super(ASAM, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        
        self.channel_attention1 = ChannelAttention(self.channel)
        self.channel_attention2 = ChannelAttention(self.channel)
        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        
        self.adaptive_pooling = AdaptivePoolingModule()
        self.conv_combine = BasicConv2d(channel * 2, channel, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        # Apply channel attention
        x1 = x1 + x1 * self.channel_attention1(x1)
        x2 = x2 + x2 * self.channel_attention2(x2)

        # Cross spatial attention
        nx1 = x1 + x1 * self.spatial_attention2(x2)
        nx2 = x2 + x2 * self.spatial_attention1(x1)

        # Combine features
        result = self.conv_combine(torch.cat([nx1, nx2], dim=1))
        result = result + x1
        
        return result, result, result


class SCEM(nn.Module):
    """Spatial Channel Enhancement Module"""
    
    def __init__(self, channel):
        super(SCEM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1)
        
        self.channel_attention1 = ChannelAttention(channel)
        self.channel_attention2 = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()
        
        self.sal_conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x1, x2):
        x2_pooled = self.sigmoid(self.max_pool(x2))
        x_weighted = x2_pooled * x1
        x = self.conv1x1(x_weighted)
        
        x_channel1 = self.channel_attention1(x) * x
        x_channel2 = self.channel_attention2(x) * x
        
        output = self.sal_conv(self.spatial_attention(x_channel1) * x_channel1) + x1 + x2_pooled + x_weighted
        
        return output, output, output


class DSFEM(nn.Module):
    """Dual-Scale Feature Enhancement Module"""
    
    def __init__(self, channel):
        super(DSFEM, self).__init__()
        self.channel = channel
        self.mixed_module = MixedModule(self.channel)
        self.progressive_module = ProgressiveModule(self.channel)
        self.multi_head_attention = MultiHeadAttention(d_model=channel)

    def forward(self, x1, x2):
        x1_enhanced = self.mixed_module(x1)
        x2_enhanced = self.progressive_module(x2)
        
        result = self.multi_head_attention(x1_enhanced, x2_enhanced, x2)
        
        return result, result, result


class MultiTaskGuidance(nn.Module):
    """Multi-task guidance module"""
    
    def __init__(self, channel):
        super(MultiTaskGuidance, self).__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                BasicConv2d(3 * channel, channel, kernel_size=3, padding=1),
                BasicConv2d(channel, channel, kernel_size=3, padding=1)
            ) for _ in range(5)
        ])

    def forward(self, x_sal, x_edg, x_ske):
        x_combined = torch.cat((x_sal, x_edg, x_ske), dim=1)
        x_output = self.conv_blocks[0](x_combined)
        return x_output


class MSPDF(nn.Module):
    """Multi-Scale Progressive Deep Fusion Network"""
    
    def __init__(self, pretrained=True, channel=32):
        super(MSPDF, self).__init__()
        self.backbone = mobilenet_v2(pretrained)

        # Translation layers
        self.translation_layer1 = Reduction(16, channel)
        self.translation_layer2 = Reduction(24, channel)
        self.translation_layer3 = Reduction(32, channel)
        self.translation_layer4 = Reduction(96, channel)
        self.translation_layer5 = Reduction(320, channel)

        # Transpose convolution layers
        self.trans_conv1 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2)
        self.trans_conv2 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2)
        self.trans_conv3 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2)
        self.trans_conv4 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2)

        # Main modules
        self.asam = ASAM(channel)
        self.scem = SCEM(channel)
        self.dsfem = DSFEM(channel)

        self.sigmoid = nn.Sigmoid()

        # Feature extractors
        self.sal_features = FeatureExtractor(channel)
        self.edg_features = FeatureExtractor(channel)
        self.ske_features = FeatureExtractor(channel)
        self.mtg = MultiTaskGuidance(channel)

        # Combination blocks
        self.sal_combine_blocks = nn.ModuleList([
            nn.Sequential(
                BasicConv2d(3 * channel, channel, kernel_size=3, padding=1),
                BasicConv2d(channel, channel, kernel_size=3, padding=1)
            ) for _ in range(5)
        ])
        self.edg_combine_blocks = nn.ModuleList([
            nn.Sequential(
                BasicConv2d(3 * channel, channel, kernel_size=3, padding=1),
                BasicConv2d(channel, channel, kernel_size=3, padding=1)
            ) for _ in range(5)
        ])
        self.ske_combine_blocks = nn.ModuleList([
            nn.Sequential(
                BasicConv2d(3 * channel, channel, kernel_size=3, padding=1),
                BasicConv2d(channel, channel, kernel_size=3, padding=1)
            ) for _ in range(5)
        ])

        # Concatenation blocks
        self.conv_cat_blocks = nn.ModuleList([
            nn.Sequential(
                BasicConv2d(2 * channel, channel, kernel_size=3, padding=1),
                BasicConv2d(channel, channel, kernel_size=3, padding=1)
            ) for _ in range(12)
        ])

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList([ConvUpsample(channel) for _ in range(12)])
        
        # Prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                BasicConv2d(channel, channel, kernel_size=3, padding=1),
                nn.Conv2d(channel, 1, kernel_size=1)
            ) for _ in range(3)
        ])

    def forward(self, x):
        size = x.size()[2:]
        
        # Backbone feature extraction
        conv1, conv2, conv3, conv4, conv5 = self.backbone(x)

        # Channel reduction
        conv1 = self.translation_layer1(conv1)
        conv2 = self.translation_layer2(conv2)
        conv3 = self.translation_layer3(conv3)
        conv4 = self.translation_layer4(conv4)
        conv5 = self.translation_layer5(conv5)

        # Progressive feature processing
        rgc5, edg5, ske5 = self.scem(conv5, conv5)
        rgc4, edg4, ske4 = self.dsfem(conv4, self.trans_conv4(conv5))
        rgc3, edg3, ske3 = self.dsfem(conv3, self.trans_conv3(conv4))
        rgc2, edg2, ske2 = self.asam(conv2, self.trans_conv2(conv3))
        rgc1, edg1, ske1 = self.asam(conv1, self.trans_conv1(conv2))

        # Feature extraction
        x_sal1, x_sal2, x_sal3, x_sal4, x_sal5 = self.sal_features(rgc1, rgc2, rgc3, rgc4, rgc5)
        x_edg1, x_edg2, x_edg3, x_edg4, x_edg5 = self.edg_features(edg1, edg2, edg3, edg4, edg5)
        x_ske1, x_ske2, x_ske3, x_ske4, x_ske5 = self.ske_features(ske1, ske2, ske3, ske4, ske5)

        # Level 5 processing
        x_sal5_n = self.sal_combine_blocks[0](torch.cat((x_sal5, x_edg5, x_ske5), 1)) + x_sal5
        x_edg5_n = self.edg_combine_blocks[0](torch.cat((x_sal5, x_edg5, x_ske5), 1)) + x_edg5
        x_ske5_n = self.ske_combine_blocks[0](torch.cat((x_sal5, x_edg5, x_ske5), 1)) + x_ske5

        # Level 4 processing
        x_sal4 = self.conv_cat_blocks[0](torch.cat((x_sal4, self.upsample_blocks[0](x_sal5_n, x_sal4)), 1))
        x_edg4 = self.conv_cat_blocks[1](torch.cat((x_edg4, self.upsample_blocks[1](x_edg5_n, x_edg4)), 1))
        x_ske4 = self.conv_cat_blocks[2](torch.cat((x_ske4, self.upsample_blocks[2](x_ske5_n, x_ske4)), 1))

        x_sal4_n = self.mtg(x_sal4, x_edg4, x_ske4) + x_sal4
        x_edg4_n = self.mtg(x_sal4, x_edg4, x_ske4) + x_edg4
        x_ske4_n = self.mtg(x_sal4, x_edg4, x_ske4) + x_ske4

# Level 3 processing
        x_sal3 = self.conv_cat_blocks[3](torch.cat((x_sal3, self.upsample_blocks[3](x_sal4_n, x_sal3)), 1))
        x_edg3 = self.conv_cat_blocks[4](torch.cat((x_edg3, self.upsample_blocks[4](x_edg4_n, x_edg3)), 1))
        x_ske3 = self.conv_cat_blocks[5](torch.cat((x_ske3, self.upsample_blocks[5](x_ske4_n, x_ske3)), 1))

        x_sal3_n = self.mtg(x_sal3, x_edg3, x_ske3) + x_sal3
        x_edg3_n = self.mtg(x_sal3, x_edg3, x_ske3) + x_edg3
        x_ske3_n = self.mtg(x_sal3, x_edg3, x_ske3) + x_ske3

        # Level 2 processing
        x_sal2 = self.conv_cat_blocks[6](torch.cat((x_sal2, self.upsample_blocks[6](x_sal3_n, x_sal2)), 1))
        x_edg2 = self.conv_cat_blocks[7](torch.cat((x_edg2, self.upsample_blocks[7](x_edg3_n, x_edg2)), 1))
        x_ske2 = self.conv_cat_blocks[8](torch.cat((x_ske2, self.upsample_blocks[8](x_ske3_n, x_ske2)), 1))

        x_sal2_n = self.mtg(x_sal2, x_edg2, x_ske2) + x_sal2
        x_edg2_n = self.mtg(x_sal2, x_edg2, x_ske2) + x_edg2
        x_ske2_n = self.mtg(x_sal2, x_edg2, x_ske2) + x_ske2

        # Level 1 processing
        x_sal1 = self.conv_cat_blocks[9](torch.cat((x_sal1, self.upsample_blocks[9](x_sal2_n, x_sal1)), 1))
        x_edg1 = self.conv_cat_blocks[10](torch.cat((x_edg1, self.upsample_blocks[10](x_edg2_n, x_edg1)), 1))
        x_ske1 = self.conv_cat_blocks[11](torch.cat((x_ske1, self.upsample_blocks[11](x_ske2_n, x_ske1)), 1))

        x_sal1_n = self.mtg(x_sal1, x_edg1, x_ske1) + x_sal1
        x_edg1_n = self.mtg(x_sal1, x_edg1, x_ske1) + x_edg1
        x_ske1_n = self.mtg(x_sal1, x_edg1, x_ske1) + x_ske1

        # Final predictions
        sal_out = self.prediction_heads[0](x_sal1_n)
        edg_out = self.prediction_heads[1](x_edg1_n)
        ske_out = self.prediction_heads[2](x_ske1_n)

        # Multi-level predictions
        x_sal2_pred = self.prediction_heads[0](x_sal2_n)
        x_edg2_pred = self.prediction_heads[1](x_edg2_n)
        x_ske2_pred = self.prediction_heads[2](x_ske2_n)
        
        x_sal3_pred = self.prediction_heads[0](x_sal3_n)
        x_edg3_pred = self.prediction_heads[1](x_edg3_n)
        x_ske3_pred = self.prediction_heads[2](x_ske3_n)

        x_sal4_pred = self.prediction_heads[0](x_sal4_n)
        x_edg4_pred = self.prediction_heads[1](x_edg4_n)
        x_ske4_pred = self.prediction_heads[2](x_ske4_n)

        x_sal5_pred = self.prediction_heads[0](x_sal5_n)
        x_edg5_pred = self.prediction_heads[1](x_edg5_n)
        x_ske5_pred = self.prediction_heads[2](x_ske5_n)
        
        # Resize all outputs to original size
        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        edg_out = F.interpolate(edg_out, size=size, mode='bilinear', align_corners=True)
        ske_out = F.interpolate(ske_out, size=size, mode='bilinear', align_corners=True)
        
        sal2 = F.interpolate(x_sal2_pred, size=size, mode='bilinear', align_corners=True)
        edg2 = F.interpolate(x_edg2_pred, size=size, mode='bilinear', align_corners=True)
        ske2 = F.interpolate(x_ske2_pred, size=size, mode='bilinear', align_corners=True)
        
        sal3 = F.interpolate(x_sal3_pred, size=size, mode='bilinear', align_corners=True)
        edg3 = F.interpolate(x_edg3_pred, size=size, mode='bilinear', align_corners=True)
        ske3 = F.interpolate(x_ske3_pred, size=size, mode='bilinear', align_corners=True)
        
        sal4 = F.interpolate(x_sal4_pred, size=size, mode='bilinear', align_corners=True)
        edg4 = F.interpolate(x_edg4_pred, size=size, mode='bilinear', align_corners=True)
        ske4 = F.interpolate(x_ske4_pred, size=size, mode='bilinear', align_corners=True)
        
        sal5 = F.interpolate(x_sal5_pred, size=size, mode='bilinear', align_corners=True)
        edg5 = F.interpolate(x_edg5_pred, size=size, mode='bilinear', align_corners=True)
        ske5 = F.interpolate(x_ske5_pred, size=size, mode='bilinear', align_corners=True)

        # Return all predictions with sigmoid activation
        return (
            x_sal1_n, 
            sal_out, self.sigmoid(sal_out), 
            edg_out, self.sigmoid(edg_out), 
            sal2, edg2, self.sigmoid(sal2), self.sigmoid(edg2), 
            sal3, edg3, self.sigmoid(sal3), self.sigmoid(edg3), 
            sal4, edg4, self.sigmoid(sal4), self.sigmoid(edg4), 
            sal5, edg5, self.sigmoid(sal5), self.sigmoid(edg5), 
            ske_out, self.sigmoid(ske_out), 
            ske2, self.sigmoid(ske2), 
            ske3, self.sigmoid(ske3), 
            ske4, self.sigmoid(ske4), 
            ske5, self.sigmoid(ske5)
        )
