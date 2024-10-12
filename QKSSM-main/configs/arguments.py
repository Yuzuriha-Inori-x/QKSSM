import argparse
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

# ======================== Dataset Configs ==========================
parser.add_argument("--audio_feat_dir", type=str, default="",
                    help="audio feat dir")
parser.add_argument("--visual_feat_dir", type=str, default="",
                    help="visual feat dir")
parser.add_argument("--qst_feat_dir", type=str, default="",
                    help="question features")
parser.add_argument("--qst_prompt_dir", type=str, default="",
                    help="question answers prompt construction")

# ======================== Label Configs ==========================
parser.add_argument("--label_train", type=str, default="../avqa_data/music_avqa/avqa-train.json",
                    help="train csv file")
parser.add_argument("--label_val", type=str, default="../avqa_data/music_avqa/avqa-val.json",
                    help="val csv file")
parser.add_argument("--label_test", type=str, default="../avqa_data/music_avqa/avqa-test.json",
                    help="test csv file")

# ======================== Model Configs ==========
parser.add_argument("--top_k", type=int, default=10, metavar='TK',
                    help="top K temporal segments")
parser.add_argument('--thread', type=float, default=0.0190,
                    help='thread for patch')

# *********************Project Layer**********************
parser.add_argument('--audio_input_dim', type=int, default=128,
                    help='preprocessed audio feature dimensions')
parser.add_argument('--visual_input_dim', type=int, default=512,
                    help='preprocessed visual feature dimensions')
parser.add_argument('--qst_input_dim', type=int, default=512,
                    help='preprocessed qeustion feature dimensions')
parser.add_argument('--qst_prompt_input_dim', type=int, default=512,
                    help='preprocessed question prompt feature dimensions')

# *********************Attention Settings**********************
parser.add_argument('--hidden_size', type=int, default=512,
                    help='Hidden layer dimension')
parser.add_argument('--dropout_p1', type=float, default=0.1,
                    help='dropout probability of attention')
parser.add_argument('--dropout_p2', type=float, default=0.1,
                    help='dropout probability of fusion')
parser.add_argument('--answer_vocab_size', type=int, default=42,
                    help='answer words number')
parser.add_argument('--cx_encoder_layers_num', type=int, default=1,
                    help='cross-attention layers')
parser.add_argument('--cx_nhead', type=int, default=4,
                    help='cross-attention heads')
parser.add_argument('--cx_d_model', type=int, default=512,
                    help='cross-attention output dimensions')
parser.add_argument('--cx_dim_feedforward', type=int, default=2048,
                    help='cross-attention hidden dimensions')

# ======================== Learning Configs ==========================
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='E',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--loss_nce_wei', type=float, default=0.2,
                    help='NCE contrastive loss weight')
parser.add_argument('--steplr_step', type=int, default=8,
                    help='after x steps it goes down')
parser.add_argument('--steplr_gamma', type=float, default=0.1,
                    help='after x steps it goes down rate')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device number')

# ======================== Save Configs ==========================
parser.add_argument("--checkpoint_file", type=str, default='TSAV',
                    help="save model name")
parser.add_argument("--model_save_dir", type=str, default='models/',
                    help="model save dir")
parser.add_argument("--mode", type=str, default='train',
                    help="with mode to use")
parser.add_argument("--result_dir", type=str, default='results/',
                    help="results files")
