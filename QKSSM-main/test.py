import torch
import torch.nn as nn


class TemporalPerception(nn.Module):
    def __init__(self, top_k):
        super(TemporalPerception, self).__init__()
        self.top_k = top_k
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)

    def forward(self, qst_input, fused_va_feat):
        qst_input = qst_input.permute(1, 0, 2)  # [1, B, C]
        fused_va_feat = fused_va_feat.permute(1, 0, 2)  # [T, B, C]
        _, temp_weights = self.attn_qst_query(qst_input, fused_va_feat, fused_va_feat,
                                              attn_mask=None, key_padding_mask=None)
        sort_index = torch.argsort(temp_weights, dim=-1)  # [B, 1, T]
        top_k_index = sort_index[:, :, -self.top_k:]  # [B, 1, Top_K]
        top_k_index_sort, indices = torch.sort(top_k_index)  # [B, 1, Top_K]
        top_k_index_sort = top_k_index_sort.cpu().numpy()  # [B, 1, Top_K],
        return top_k_index_sort


class TopKSegsSelection(nn.Module):
    def __init__(self, top_k):
        super(TopKSegsSelection, self).__init__()
        self.top_k = top_k

    def forward(self, top_k_index_sort, patch_feat, audio_feat):
        B, T, N, C = patch_feat.shape
        output_patch_feat = torch.zeros(B, self.top_k, N, C)
        output_audio_feat = torch.zeros(B, self.top_k, C)

        for batch_idx in range(B):
            idx = 0
            for temp_idx in top_k_index_sort.tolist()[batch_idx][0]:
                output_patch_feat[batch_idx, idx, :, :] = patch_feat[batch_idx, temp_idx, :, :]
                output_audio_feat[batch_idx, idx, :] = audio_feat[batch_idx, temp_idx, :]
                idx += 1

        return output_patch_feat, output_audio_feat


if __name__ == '__main__':
    temp = TemporalPerception(top_k=20)
    ssl = TopKSegsSelection(top_k=20)
    qst_input = torch.rand(2, 1, 512)
    fused_va_feat = torch.rand(2, 60, 512)
    patch_feat = torch.rand(2, 60, 49, 512)
    audio_feat = torch.rand(2, 60, 512)
    top_k_index_sort = temp(qst_input, fused_va_feat)
    print(top_k_index_sort.shape)
    patch_feat_tss, audio_feat_tss = ssl(top_k_index_sort, patch_feat, audio_feat)
    print(patch_feat_tss.shape, audio_feat_tss.shape)
