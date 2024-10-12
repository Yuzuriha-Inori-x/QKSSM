import torch
import torch.nn as nn
from torch.autograd import Variable
from loss import AVContrastive_loss_50 as Contrastive_loss
from fusion import AttFlat
from transformer_encoder import SAEncoder, CXEncoder


class TemporalPerception(nn.Module):
    def __init__(self, args):
        super(TemporalPerception, self).__init__()
        self.args = args
        self.attn_qst_query_visual = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.attn_qst_query_audio = nn.MultiheadAttention(512, 4, dropout=0.1)

    def forward(self, qst_prompt, visual_feat, audio_feat, modality_wei):
        qst_input = qst_prompt.permute(1, 0, 2)  # [1, B, C]
        visual_feat = visual_feat.permute(1, 0, 2)  # [T, B, C]
        audio_feat = audio_feat.permute(1, 0, 2)  # [T, B, C]
        _, temp_weights_visual = self.attn_qst_query_visual(qst_input, visual_feat, visual_feat,
                                                            attn_mask=None, key_padding_mask=None)
        _, temp_weights_audio = self.attn_qst_query_audio(qst_input, audio_feat, audio_feat,
                                                          attn_mask=None, key_padding_mask=None)
        temp_weights = temp_weights_visual * modality_wei[:, 0].unsqueeze(-1).unsqueeze(-1) + \
                       temp_weights_audio * modality_wei[:, 1].unsqueeze(-1).unsqueeze(-1)
        sort_index = torch.argsort(temp_weights, dim=-1)  # [B, 1, T]
        top_k_index = sort_index[:, :, -self.args.top_k:]  # [B, 1, Top_K]
        top_k_index_sort, indices = torch.sort(top_k_index)  # [B, 1, Top_K]
        top_k_index_sort = top_k_index_sort.cpu().numpy()  # [B, 1, Top_K],
        return top_k_index_sort


class TopKSegsSelection(nn.Module):
    def __init__(self, args):
        super(TopKSegsSelection, self).__init__()
        self.args = args

    def forward(self, top_k_index_sort, patch_feat, audio_feat):
        B, T, N, C = patch_feat.shape
        output_patch_feat = torch.zeros(B, self.args.top_k, N, C).cuda()
        output_audio_feat = torch.zeros(B, self.args.top_k, C).cuda()

        for batch_idx in range(B):
            idx = 0
            for temp_idx in top_k_index_sort.tolist()[batch_idx][0]:
                output_patch_feat[batch_idx, idx, :, :] = patch_feat[batch_idx, temp_idx, :, :]
                output_audio_feat[batch_idx, idx, :] = audio_feat[batch_idx, temp_idx, :]
                idx += 1

        return output_patch_feat, output_audio_feat


class TSAV(nn.Module):
    def __init__(self,
                 args,
                 audio_input_dim=128,
                 visual_input_dim=512,
                 qst_input_dim=512,
                 qst_prompt_input_dim=512,
                 hidden_size=512,
                 answer_vocab_size=42,
                 dropout_p1=0.1,
                 dropout_p2=0.1,
                 cx_encoder_layers_num=1,
                 cx_nhead=4,
                 cx_d_model=512,
                 cx_dim_feedforward=2048,
                 top_k=10,
                 thread=0.0190
                 ):
        super(TSAV, self).__init__()
        self.args = args
        self.top_k = top_k
        self.thread = thread
        self.hidden_size = hidden_size

        # Project Layers
        self.fc_audio = nn.Linear(audio_input_dim, hidden_size)
        self.fc_visual = nn.Linear(visual_input_dim, hidden_size)
        self.fc_visual_patch = nn.Linear(visual_input_dim, hidden_size)
        self.fc_word = nn.Linear(qst_input_dim, hidden_size)
        self.fc_sentence = nn.Linear(qst_input_dim, hidden_size)
        self.fc_qst_prompt = nn.Linear(qst_prompt_input_dim, hidden_size)

        # Modules
        self.TemporalPerception = TemporalPerception(args)
        self.TopKSegsSelection = TopKSegsSelection(args)

        self.audio_qst_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                           dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)
        self.patch_qst_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                           dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)

        self.qst_audio_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                           dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)
        self.qst_patch_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num,
                                           dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)

        self.wei_from_qst = nn.Linear(hidden_size, 2)
        self.attflat = AttFlat(hidden_size, hidden_size, 1, answer_vocab_size, dropout_r=dropout_p2)

    def gene_question_as_key_pad_mask(self, q_feat, seq_length):
        mask = torch.ones(q_feat.shape[:2])
        for i, l in enumerate(seq_length):
            mask[i][l:] = 0
        mask = mask.to(torch.bool)
        mask = ~mask
        return mask

    def make_mask(self, seq, seq_length):
        mask = torch.ones(seq.shape[:2])
        for i, l in enumerate(seq_length):
            mask[i][l:] = 0
        mask = Variable(mask)
        mask = mask.to(torch.float)
        return mask

    def forward(self, audio_input, visual_input, visual_patch_input, qst_input, qst_len, qst_prompt_input):
        """
        :param audio_input: from vggish [b, t, c=128]
        :param visual_input: from clip-vision-encoder [b ,t, c=512]
        :param visual_patch_input: from clip-vision-encoder [b ,t, n=49, c=512]
        :param qst_input: from clip-text-encoder [b, n=77, c=512]
        :param qst_len: [b, 1]
        :param qst_prompt_input: from clip-text-encoder [b, c=512]
        :return:
        """
        # * visual patch
        patch_feat = self.fc_patch(visual_patch_input)  # [B, T, NUM, C]
        # * visual
        visual_feat = self.fc_visual(visual_input)  # [B, T, C]
        # * audio
        audio_feat = self.fc_audio(audio_input)  # [B, T, C]
        # * word level
        word_feat = qst_input[:, 1:, :]  # [B, 76, 512]
        word_feat = self.fc_word(word_feat)  # [B, 76, 512]
        # * sentence level
        sentence_feat = qst_input[:, 0, :]  # [B, 512]
        sentence_feat = self.fc_sentence(sentence_feat)  # [B, 512]
        qst_temp = sentence_feat.unsqueeze(1)  # [B, 1, C]
        qst_repeat = qst_temp.repeat(1, self.top_k_frames, 1)  # [B, T, C]
        # * qst prompt
        qst_prompt = self.fc_qst_prompt(qst_prompt_input).unsqueeze(1)  # [B, 1, C]

        ################################################################################################################
        # * modality-aware weights
        modality_wei = self.wei_from_qst(sentence_feat)
        modality_wei = torch.softmax(modality_wei, dim=-1)

        # * temporal perception
        top_k_index_sort = self.TemporalPerception(qst_prompt, visual_feat, audio_feat, modality_wei)  # [B, 1, Top_K]

        patch_feat_tss, audio_feat_tss = self.TopKSegsSelection(top_k_index_sort, patch_feat, audio_feat)
        # [B, Top_K, N, C], [B, Top_K, C]

        ################################################################################################################
        # * auido-visual clues mining
        question_mask = self.gene_question_as_key_pad_mask(word_feat, ques_len)

        B, T, PATCH_NUM, _ = patch_feat_tss.shape  # [B, Top_K, NUM, C]
        patch_feat_tss = patch_feat_tss.view(B, T * PATCH_NUM, -1)  # [B, Top_K*NUM, C]

        cx_patch_feat, _, _ = self.patch_qst_encoder(patch_feat_tss.permute(1, 0, 2), word_feat.permute(1, 0, 2),
                                                     attn_mask=None, key_padding_mask=question_mask, q_pos_embed=None,
                                                     k_pos_embed=None)  # [Top_K*N, B, 512]
        cx_audio_feat, _, _ = self.audio_qst_encoder(audio_feat_tss.permute(1, 0, 2), word_feat.permute(1, 0, 2),
                                                     attn_mask=None, key_padding_mask=question_mask, q_pos_embed=None,
                                                     k_pos_embed=None)  # [Top_K, B, 512]

        ################################################################################################################
        # adaptive contrastive losses.
        CL_loss_PQ = Contrastive_loss(cx_patch_feat.permute(1, 0, 2), qst_repeat, thread=self.thread)
        CL_loss_PA = Contrastive_loss(cx_patch_feat.permute(1, 0, 2), cx_audio_feat.permute(1, 0, 2),
                                      thread=self.thread)
        CL_loss = CL_loss_PQ + CL_loss_PA

        ################################################################################################################
        # * question-aware audio-visual fusion
        cx_patch_feat2, _, _ = self.qst_patch_encoder(word_feat.permute(1, 0, 2), cx_patch_feat, attn_mask=None,
                                                      key_padding_mask=None, q_pos_embed=None, k_pos_embed=None)
        # [cur_max_lenth, B, 512]
        cx_audio_feat2, _, _ = self.qst_audio_encoder(word_feat.permute(1, 0, 2), cx_audio_feat, attn_mask=None,
                                                      key_padding_mask=None, q_pos_embed=None, k_pos_embed=None)
        # [cur_max_lenth, B, 512]
        cx_patch_feat2 = cx_patch_feat2.permute(1, 0, 2)  # [B, cur_max_lenth, 512]
        cx_audio_feat2 = cx_audio_feat2.permute(1, 0, 2)  # [B, cur_max_lenth, 512]

        # * modality-aware fusion
        cx_patch_feat2 = cx_patch_feat2 * modality_wei[:, 0].unsqueeze(-1).unsqueeze(-1)
        cx_audio_feat2 = cx_audio_feat2 * modality_wei[:, 1].unsqueeze(-1).unsqueeze(-1)
        cx_fused_feat = cx_patch_feat2 + cx_audio_feat2  # [B, cur_max_lenth, 512]

        ################################################################################################################
        # answer prediction
        q_mask = self.make_mask(word_feat, qst_len)  # [B, 14]
        fusion_out = self.attflat(cx_fused_feat, q_mask)
        return fusion_out, CL_loss
