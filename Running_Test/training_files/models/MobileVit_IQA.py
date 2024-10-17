import torch
import torch.nn as nn
import timm

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Attention_Block(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x

class Self_Attention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()

        self.qConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.kConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.vConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inFeature):
        bs, C, w, h = inFeature.size()

        proj_query = self.qConv(inFeature).view(bs, -1, w * h).permute(0, 2, 1)
        proj_key = self.kConv(inFeature).view(bs, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.vConv(inFeature).view(bs, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, w, h)

        out = self.gamma * out + inFeature

        return out


class MAL(nn.Module):
    """
        Multi-view Attention Learning (MAL) module
    """

    def __init__(self, in_dim=768, feature_num=4, feature_size=28):
        super().__init__()

        self.channel_attention = Attention_Block(in_dim * feature_num)  # Channel-wise self attention
        self.feature_attention = Attention_Block(feature_size ** 2 * feature_num)  # Pixel-wise self attention

        # Self attention module for each input feature
        self.attention_module = nn.ModuleList()
        for _ in range(feature_num):
            self.attention_module.append(Self_Attention(in_dim))

        self.feature_num = feature_num
        self.in_dim = in_dim

    def forward(self, features):
        feature = torch.tensor([]).cuda()
        for index, _ in enumerate(features):
            feature = torch.cat((feature, self.attention_module[index](features[index]).unsqueeze(0)), dim=0)
        features = feature

        input_tensor = rearrange(features, 'n b c w h -> b (n c) (w h)')  # bs, 768 * feature_num, 28 * 28
        bs, _, _ = input_tensor.shape  # [2, 3072, 784]

        in_feature = rearrange(input_tensor, 'b (w c) h -> b w (c h)', w=self.in_dim, c=self.feature_num)  # bs, 768, 28 * 28 * feature_num
        feature_weight_sum = self.feature_attention(in_feature)  # bs, 768, 768

        in_channel = input_tensor.permute(0, 2, 1)  # bs, 28 * 28, 768 * feature_num
        channel_weight_sum = self.channel_attention(in_channel)  # bs, 28 * 28, 28 * 28

        weight_sum_res = (rearrange(feature_weight_sum, 'b w (c h) -> b (w c) h', w=self.in_dim,
                                    c=self.feature_num) + channel_weight_sum.permute(0, 2, 1)) / 2  # [2, 3072, 784]

        weight_sum_res = torch.mean(weight_sum_res.view(bs, self.feature_num, self.in_dim, -1), dim=1)

        return weight_sum_res  # bs, 768, 28 * 28

class Local_Distortion_Aware(nn.Module):
    """
        Multi-view Attention Learning (MAL) module
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_dim, out_dim * 2, 1, 1)
        self.grelu = nn.GELU()
        self.avg2 = nn.AvgPool2d(4, 4)
        self.cnn2 = nn.Conv2d(out_dim * 2, out_dim, 1, 1)
        self.avg = nn.AdaptiveAvgPool2d((22, 22))

    def forward(self, features):
        local_1 = self.avg(self.grelu(self.cnn1(features)))
        # print('local_1', local_1.shape)
        local_2 = self.cnn2(local_1)
        # print('local_2', local_2.shape) # bs, 128, 16, 16

        return local_2.unsqueeze(1)  # bs, 1, 128, 16, 16
    
class Model(nn.Module):
    def __init__(self, drop=0.1, dim_mlp=768, is_teacher=False):
        super().__init__()
        self.input_size = 16
        self.dim_mlp = dim_mlp

        out_indices = [0, 1, 2, 3, 4]
        self.global_vit = timm.create_model('mobilevitv2_200', features_only=True, out_indices=out_indices, pretrained=False)
        
        self.LGF1 = Local_Distortion_Aware(128, 256)
        self.LGF2 = Local_Distortion_Aware(256, 256)
        self.LGF3 = Local_Distortion_Aware(512, 256)
        self.LGF4 = Local_Distortion_Aware(768, 256)
        self.LGF5 = Local_Distortion_Aware(1024, 256)
        
        self.MALs = nn.ModuleList()
        for _ in range(3):
            self.MALs.append(MAL(256, feature_num=5, feature_size=22))

        # Image Quality Score Regression
        dim_mlp = 256
        self.fusion_mal = MAL(256, feature_num=3, feature_size=22)
        self.cnn = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((4, 4)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_score = nn.Sequential(
            nn.Linear(128, 128 // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128 // 2, 1),
        )
        self.input_size = 22
        self.is_teacher = is_teacher

    def forward(self, full_img):
        
        global_features = self.global_vit(full_img)
        global_feature_list = None
        for idx, _ in enumerate(global_features):
            if global_feature_list is None:
                global_feature_list = getattr(self, 'LGF{}'.format(idx + 1))(global_features[idx]) # bs 128
            else:
                global_feature_list = torch.cat((global_feature_list, getattr(self, 'LGF{}'.format(idx + 1))(global_features[idx])), dim=1)

        x = global_feature_list # bs, 5, 256, 16, 16
        x = x.permute(1, 0, 2, 3, 4)  # bs, 4, 768, 28 * 28

        # Different Opinion Featurs (DOF)
        DOF = torch.tensor([]).cuda()
        for index, _ in enumerate(self.MALs):
            DOF = torch.cat((DOF, self.MALs[index](x).unsqueeze(0)), dim=0)
        DOF = rearrange(DOF, 'n c d (w h) -> n c d w h', w=self.input_size, h=self.input_size)  # M, bs, 768, 28, 28

        # Image Quality Score Regression
        fusion_mal = self.fusion_mal(DOF).permute(0, 2, 1)  # bs, 28 * 28 768
        IQ_feature = fusion_mal.permute(0, 2, 1) # bs, 768, 28 * 28
        IQ_feature = rearrange(IQ_feature, 'c d (w h) -> c d w h', w=self.input_size, h=self.input_size) # bs, 768, 28, 28
        score = self.cnn(IQ_feature).squeeze(-1).squeeze(-1)
        score = self.fc_score(score).view(-1)
        
        if self.is_teacher:
            return x, DOF, score
        return score