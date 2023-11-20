
import copy
from Code.material.LR_Transformer_utils import ResMlp,Attention,PatchEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.material.attention_utils import SpatialAttention,ChannelAttention
from Code.material.backbone.Res2Net import res2net50_v1b_26w_4s
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':  # True
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':  # False
            width_pad_size = 0
            height_pad_size = 0
        else:  # False
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)  # 0
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)  # 0
        pad_size = (width_pad_size, height_pad_size)  # pad_size:(0,0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups,bias=bias)
        self.reset_parameters()

        if bn is True:  # True
            self.bn = nn.BatchNorm2d(out_channels)  # out_channels:32
        else:
            self.bn = None

        if relu is True:  # False
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode # mode='h' | mode:'w'

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

def swish(x):
    return x * torch.sigmoid(x)

# Activation function
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,kernel_size=kernel_size, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(SA_kernel, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x

class SA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SA, self).__init__()

        self.branch0 = conv(in_channel, out_channel, kernel_size=1)
        self.branch1 = SA_kernel(in_channel, out_channel,receptive_size=3)
        self.branch2 = SA_kernel(in_channel, out_channel, receptive_size=5)
        self.branch3 = SA_kernel(in_channel, out_channel, receptive_size=7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

        self.relu = nn.ReLU(True)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x

# multi-scale feature fusion module
class MFF(nn.Module):

    def __init__(self, channel):
        super(MFF, self).__init__()
        self.conv1 = conv(channel*3 ,channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 768, 3, bn=False)

        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')
        self.Channel_attention =  ChannelAttention(in_planes=32)
        self.Spatial_attention = SpatialAttention(kernel_size=7)
        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

        self.fc = nn.Sequential(

            nn.Conv2d(in_channels=32,out_channels=16,padding=1,kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16,out_channels=32,padding=1,kernel_size=3,stride=1),
            nn.Sigmoid()
        )
        self.batchNorm = nn.BatchNorm2d(num_features=32,eps=1e-5,momentum=0.2)
        pass

    def forward(self, f1, f2, f3):
        f1 = self.upsample(f1, f3.shape[-2:])
        f2 = self.upsample(f2, f3.shape[-2:])

        f1 = self.Channel_attention(f1)
        f2 = self.Spatial_attention(f2)
        f1 = self.upsample(f1, f3.shape[-2:])
        f2 = self.upsample(f2, f3.shape[-2:])
        f3 = torch.cat([f1, f2, f3], dim=1)
        f3 = self.conv1(f3)

        Hf3 = self.Hattn(f3)
        Wf3 = self.Wattn(f3)

        f3 = self.conv2(Hf3 + Wf3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)

        b, c, W, H = f3.size()  # b:24,c:32 H:44 W:44
        f3 = torch.sigmoid(f3)

        y = self.fc(f3)
        y = y.view(b, c, W, H)

        y = self.batchNorm(y)
        y = f3 * y

        out = self.conv5(y)
        return out


class StackTransformer(nn.Module):
    def __init__(self,c):

        super(StackTransformer, self).__init__()

        self.layer = nn.ModuleList() # 创建了一个名为layer的nn.ModuleList对象，用于存储多个Transformer_Block模块
        self.encoder_norm = LayerNorm(768, eps=1e-6) # 对输入数据进行标准化处理
        for _ in range(24):
            layer = Transformer_Block()
            self.layer.append(copy.deepcopy(layer)) # 将刚创建的layer模块添加到self.layer列表中。这里使用copy.deepcopy()来创建layer的深层拷贝，以确保每个模块都是独立的。

    def forward(self, hidden_states):

        for layer_block in self.layer:
            # 对hidden_states应用layer_block模块，并将返回的新的hidden_states和权重（weights）存储起来。
            hidden_states, weights = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded

class Transformer_Block(nn.Module):
    def __init__(self):
        super(Transformer_Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = ResMlp()
        self.attn = Attention()


    def forward(self, x):
        if x.size()[1] == 768 and x.size()[2] == 196:
            x = x.permute(0, 2, 1)
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

import math
def positional_embedding(length, hidden_dim):
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) * -(math.log(10000.0) / hidden_dim))
    pos_enc = torch.zeros((length, hidden_dim))

    # Calculate the sine and cosine components of positional encoding
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc


class CST_Net(nn.Module):
    def __init__(self, channel=32, n_class=1):
        super(CST_Net, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.sa_1 = SA(512, channel) # channel：32
        self.sa_2 = SA(1024, channel)
        self.sa_3 = SA(2048, channel)

        # ---- dim_768_to_1 ----
        self.dim_768_to_1 = nn.Conv2d(768,1,kernel_size=1,stride=1)
        # self.dim_768_to_3 = nn.Conv2d(768,3,1,1)
        # ---- StackTransformer -----
        self.pos_embed = nn.Parameter(positional_embedding(14 * 14, 768))
        self.pos_drop = nn.Dropout(p=0.0)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.patch_embedding = PatchEmbedding(in_channels=768, patch_size=1, emb_size=768)
        self.transformer = StackTransformer(channel)

        # ---- multi-scale feature fusion module----
        self.MFF = MFF(channel)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256+64, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, n_class, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64+64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64+64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)

        # ---- edge branch ----
        # self.edge_conv1_x1_256_to_64 = BasicConv2d(256, 64, kernel_size=1)
        self.boundary_conv1_x2_rbf_32_to_64 = BasicConv2d(32, 64, kernel_size=1)
        self.boundary_supervise_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.boundary_supervise_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.boundary_supervise_4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)


        # ----- contour -----
        # self.contour = nn.Conv2d(1, 3, kernel_size=1)
    def forward(self, x):
        # B:batch_size H:height W:width
        B ,channel,H,W= x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # ---- low-level features ----
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)

        # ---- high-level features ----
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # ---- use SA ----
        sa_1 = self.sa_1(x2)
        sa_2 = self.sa_2(x3)
        sa_3 = self.sa_3(x4)


        x = self.boundary_conv1_x2_rbf_32_to_64(sa_1) # [24, 32, 32, 32]

        x = self.boundary_supervise_2(x)
        boundary_guidance = self.boundary_supervise_3(x)
        boundary_lateral = self.boundary_supervise_4(boundary_guidance)
        boundary_lateral = F.interpolate(boundary_lateral,scale_factor=8,mode='bilinear')

        # ---- multi-scale feature fusion module----
        ra5_feat = self.MFF(sa_3, sa_2, sa_1)

        # ---- StackTransformer ---
        ra5_feat = F.interpolate(ra5_feat,size=(14,14),mode='bilinear')

        ra5_feat = ra5_feat.view(B,196,-1)
        ra5_feat = self.pos_drop(ra5_feat + self.pos_embed)
        ra5_feat = ra5_feat.view(B, 768, 14, -1)
        #
        # ra5_feat = ra5_feat.view(B, 196, -1)
        # ra5_feat = self.pos_drop(ra5_feat + self.pos_embed)
        # ra5_feat = ra5_feat.view(B, 768, 14, 14)
        ra5_feat = self.patch_embedding(ra5_feat)
        ra5_feat  = ra5_feat.view(B,768,196)

        ra5_feat = self.transformer(ra5_feat)
        ra5_feat = ra5_feat.permute(0,2,1)
        ra5_feat = ra5_feat.view(B,768,14,14)

        ra5_feat = self.dim_768_to_1(ra5_feat) # (24,1,14,14)
        if H == 256:
            ra5_feat = F.interpolate(ra5_feat,size=(32,32),mode='bilinear')
            lateral_map_5 = F.interpolate(ra5_feat, size=(256, 256), mode='bilinear')  # [24,1,256,256]
        elif H == 352:
            ra5_feat = F.interpolate(ra5_feat, size=(44, 44), mode='bilinear')
            lateral_map_5 = F.interpolate(ra5_feat, size=(352, 352), mode='bilinear')  # [24,1,352,352]
        elif H == 448:
            ra5_feat = F.interpolate(ra5_feat, size=(56,56), mode='bilinear')
            lateral_map_5 = F.interpolate(ra5_feat, size=(448,448),mode='bilinear')  # [24,1,448,448]

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        # --- the current prediction is upsampled from its deeper layer ---
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)

        x = torch.cat((self.ra4_conv1(x), F.interpolate(boundary_guidance, scale_factor=1/4, mode='bilinear')), dim=1)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32,mode='bilinear')

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)

        x = torch.cat((self.ra3_conv1(x), F.interpolate(boundary_guidance, scale_factor=1/2, mode='bilinear')), dim=1)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x,scale_factor=16,mode='bilinear')

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = torch.cat((self.ra2_conv1(x), F.interpolate(boundary_guidance, scale_factor=1, mode='bilinear')), dim=1)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x,scale_factor=8, mode='bilinear') # [1,1,352,352]

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, boundary_lateral



if __name__ == '__main__':
    device= 'cuda:1'
    ras = CST_Net().to(device)
    input_tensor = torch.randn(24, 3, 352, 352).to(device)

    out = ras(input_tensor)
    print(out[0].shape)
    # print(ras) # 打印网络结构
    import thop

    # 创建一个示例输入张量
    input_tensor = torch.randn(1, 3, 352, 352).to(device)

    # 使用thop来估算FLOP
    flops, params = thop.profile(ras, inputs=(input_tensor,))

    print(f"FLOPs: {flops / 1e9} G FLOPs")  # 打印出FLOP的结果，以十亿（G）为单位

