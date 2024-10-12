import numpy as np
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import ast
import json


class AVQA_dataset(Dataset):
    def __init__(self, label,
                 audios_feat_dir, visual_feat_dir,
                 qst_prompt_dir, qst_feat_dir,
                 transform=None, mode_flag='train'):
        samples = json.load(open('../avqa_data/music_avqa/avqa-train.json', 'r'))

        # * Question
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1
            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.ans_vocab = ans_vocab
        self.ans_to_idx = {id: index for index, id in enumerate(self.ans_vocab)}

        # loading train/val/test json file.
        self.samples = json.load(open(label, 'r'))
        self.max_len = 14  # question length

        self.audios_feat_dir = audios_feat_dir
        self.visual_feat_dir = visual_feat_dir

        self.qst_prompt_dir = qst_prompt_dir
        self.qst_feat_dir = qst_feat_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample['video_id']

        # * audio vggish
        audios_feat = np.load(os.path.join(self.audios_feat_dir, name + '.npy'))  # [60, 128]
        audio_feat = audios_feat[:60, :]
        audio_feat = torch.from_numpy(audio_feat).float()

        # * visual clip
        visual_frame_feat = np.load(os.path.join(self.visual_feat_dir, name + '.npy'))  # [60, 50, 512]
        visual_feat = visual_frame_feat[:60, 0, :]
        visual_feat = torch.from_numpy(visual_feat).float()

        # * patch clip
        visual_patch_feat = visual_frame_feat[:60, 1:, :]  # [60, 49, 512]
        visual_patch_feat = torch.from_numpy(visual_patch_feat).float()

        # * question
        question_id = int(sample['question_id'])
        question_feat = np.load(os.path.join(self.qst_feat_dir, str(question_id) + '.npy'))[0]  # [1, 77, 512]
        question_feat = question_feat[0]  # [77, 512]
        question_feat = torch.from_numpy(question_feat).float()

        # * question length
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]
        ques_len = len(question)
        ques_len = torch.from_numpy(np.array(ques_len))  # question length.

        # * question prompt
        question_prompt = np.load(os.path.join(self.qst_prompt_dir, str(question_id) + '.npy'))
        question_prompt = question_prompt[0, 0, :]
        question_prompt = torch.from_numpy(question_prompt).float()

        # * answer
        answer = sample['anser']
        answer_label = self.ans_to_idx[answer]
        answer_label = torch.from_numpy(np.array(answer_label)).long()

        sample = {
            'video_name': name,
            'audio_feat': audio_feat,
            'visual_feat': visual_feat,
            'visual_patch_feat': visual_patch_feat,
            'question_feat': question_feat,
            'question_len': ques_len,
            'question_prompt': question_prompt,
            'answer_label': answer_label,
            'question_id': question_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        video_name = sample['video_name']
        audio_feat = sample['audio_feat']
        visual_feat = sample['visual_feat']
        visual_patch_feat = sample['visual_patch_feat']
        question_feat = sample['question_feat']
        question_len = sample['question_len']
        question_prompt = sample['question_prompt']
        answer_label = sample['answer_label']
        question_id = sample['question_id']

        return {'video_name': video_name,
                'audio_feat': audio_feat,
                'visual_feat': visual_feat,
                'visual_patch_feat': visual_patch_feat,
                'question_feat': question_feat,
                'question_len': question_len,
                'question_prompt': question_prompt,
                'answer_label': answer_label,
                'question_id': question_id}
