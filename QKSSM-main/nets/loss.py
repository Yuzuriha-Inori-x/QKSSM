import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def AVContrastive_loss_100(obj_feature: torch.Tensor(), audio_feature: torch.Tensor(), tau=0.4, thread=0.0110):
    norm_e = 1e-3
    B, T, d = audio_feature.shape

    x = obj_feature.reshape(B, T, -1, d)
    y = audio_feature.reshape(B, T, 1, d)
    obj_feature = obj_feature.reshape(B * T, -1, d)
    audio_feature = audio_feature.reshape(B * T, 1, d)

    z1 = obj_feature
    z2 = audio_feature
    cos_matrix = F.cosine_similarity(z1, z2, dim=2, eps=1e-6).reshape(B, T, -1)
    cos_matrix_F = F.softmax(cos_matrix, dim=-1)

    cos_matrix_exp = torch.exp(cos_matrix_F / tau)

    # thread = 0.0110
    zero = torch.zeros_like(cos_matrix)
    positive_matrix = torch.where(cos_matrix_F < thread, zero, cos_matrix_exp)

    second = torch.sum(cos_matrix_exp, dim=-1)
    first = torch.sum(positive_matrix, dim=-1)
    first[first == 0] = norm_e

    other_neg_list = []
    for i in range(B):
        obj = torch.cat((x[:i, :, :, :], x[i + 1:, :, :, :]), dim=0)
        aud = y[i, :, :, :]
        cos_mat_other = F.cosine_similarity(obj, aud, dim=-1)
        other_neg_list.append(cos_mat_other.unsqueeze(0))

    neg_01 = torch.cat(other_neg_list, dim=0)
    neg_02 = torch.mean(neg_01, dim=1)
    neg_03 = torch.mean(neg_02, dim=-1)

    res = torch.mean((1 / T) * torch.sum(-torch.log(first / (second + neg_03)), dim=-1))

    return res if res > norm_e else norm_e


def AVContrastive_loss_50(obj_feature: torch.Tensor(), audio_feature: torch.Tensor(), thread=0.0210, tau=0.4):
    norm_e = 1e-3
    B, T, d = audio_feature.shape

    x = obj_feature.reshape(B, T, -1, d)  # [B, T, NUM, D]
    y = audio_feature.reshape(B, T, 1, d)  # [B, T, 1, D]
    obj_feature = obj_feature.reshape(B * T, -1, d)  # [B * T, NUM, D]
    audio_feature = audio_feature.reshape(B * T, 1, d)  # [B * T, 1, D]

    z1 = obj_feature  # [B * T, NUM, D]
    z2 = audio_feature  # [B * T, 1, D]
    cos_matrix = F.cosine_similarity(z1, z2, dim=2, eps=1e-6).reshape(B, T, -1)  # [B, T, NUM]
    cos_matrix_F = F.softmax(cos_matrix, dim=-1)  # [B, T, NUM]

    cos_matrix_exp = torch.exp(cos_matrix_F / tau)  # [B, T, NUM]

    # thread = 0.0210
    zero = torch.zeros_like(cos_matrix)  # [B, T, NUM]
    positive_matrix = torch.where(cos_matrix_F < thread, zero, cos_matrix_exp)  # [B, T, NUM]

    second = torch.sum(cos_matrix_exp, dim=-1)  # [B, T]
    first = torch.sum(positive_matrix, dim=-1)  # [B, T]
    first[first == 0] = norm_e  # [B, T]

    other_neg_list = []
    for i in range(B):
        obj = torch.cat((x[:i, :, :, :], x[i + 1:, :, :, :]), dim=0)
        aud = y[i, :, :, :]
        cos_mat_other = F.cosine_similarity(obj, aud, dim=-1)
        other_neg_list.append(cos_mat_other.unsqueeze(0))

    neg_01 = torch.cat(other_neg_list, dim=0)  # [B, 1, T, NUM]
    neg_02 = torch.mean(neg_01, dim=1)  # [B, T, NUM]
    neg_03 = torch.mean(neg_02, dim=-1)  # [B, T]
    res = torch.mean((1 / T) * torch.sum(-torch.log(first / (second + neg_03)), dim=-1))

    return res if res > norm_e else norm_e


def FNAContrastive_loss(img, aud, tau=0.03, high_conf_thresh=0.6):
    """
    :param img: [B, T*NUM, C]
    :param aud: [B, T, C]
    :param tau:
    :param high_conf_thresh:
    :return:
    """

    b1, t1, c1 = aud.size()
    aud = nn.functional.normalize(aud, dim=1)

    img = rearrange(img, 'b (t num) c -> (b t) c num', b=b1, t=t1, c=c1)
    img = img.view(b1, t1, c1, 7, 7)
    img = nn.functional.normalize(img, dim=1)

    b = img.shape[0]
    t = img.shape[1]
    c = img.shape[2]
    h = img.shape[3]
    w = img.shape[4]

    B = b * t

    img = img.contiguous().view(b * t, c, h, w)
    aud = aud.contiguous().view(b * t, c)

    aud_attn = (aud @ aud.transpose(0, 1)) / tau
    # print('aud_attn:', aud_attn.shape)

    img_avg = nn.AdaptiveAvgPool2d((1, 1))(img)[:, :, 0, 0]
    # print('img_avg:', img_avg.shape)
    img_attn = (img_avg @ img_avg.transpose(0, 1)) / tau
    # print('img_attn:', img_attn.shape)

    Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / tau
    # print('Slogits:', Slogits.shape)

    loc_map = Slogits[torch.arange(B), torch.arange(B)]
    # print('loc_map:', loc_map.shape)
    loc_map = (loc_map - torch.amin(loc_map, (1, 2), keepdim=True)) / \
              (torch.amax(loc_map, (1, 2), keepdim=True) - torch.amin(loc_map, (1, 2), keepdim=True) + 1e-5)
    # print('loc_map:', loc_map.shape)

    frg_feature = img * (loc_map > high_conf_thresh).unsqueeze(1)  # foreground visual features
    # print('frg_feature:', frg_feature.shape)
    frg_feature = frg_feature.flatten(-2, -1).mean(dim=-1)
    # print('frg_feature:', frg_feature.shape)
    frg_attn = (frg_feature @ frg_feature.transpose(0, 1)) / tau
    # print('frg_attn:', frg_attn.shape)

    logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
    # print('logits:', logits.shape)

    fnac_loss1 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(logits, dim=1))  # FNS-1
    fnac_loss2 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(frg_attn, dim=1))  # TNS
    fnac_loss3 = F.l1_loss(torch.softmax(img_attn, dim=1), torch.softmax(logits, dim=1))  # FNS-2

    return fnac_loss1, fnac_loss2, fnac_loss3


if __name__ == "__main__":
    obj_feature = torch.randn(2, 490, 512)
    audio_feature = torch.randn(2, 10, 512)
    loss = AVContrastive_loss_50(obj_feature, audio_feature)
    fnac_loss1, fnac_loss2, fnac_loss3 = FNAContrastive_loss(obj_feature, audio_feature)
    print(loss, (fnac_loss1 + fnac_loss2 + fnac_loss3))
    print(fnac_loss1, fnac_loss2, fnac_loss3)
