import sys

sys.path.append("../TSAV-main")
import warnings
import random
from datetime import datetime
from torchvision import transforms
from dataloader import *
from nets.TSAV import TSAV
from configs.arguments import parser

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')

print("\n--------------- TSAV --------------- \n")


def model_test(model, test_loader, test_json_file):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open(test_json_file, 'r'))

    # useing index of question
    questionid_to_samples = {}
    for sample in samples:
        ques_id = sample['question_id']
        if ques_id not in questionid_to_samples.keys():
            questionid_to_samples[ques_id] = sample
        else:
            print("question_id_duplicated:", ques_id)

    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            audio_feat = sample['audio_feat'].to('cuda')
            visual_feat = sample['visual_feat'].to('cuda')
            visual_patch_feat = sample['visual_patch_feat'].to('cuda')
            question_feat = sample['question_feat'].to('cuda')
            question_len = sample['question_len'].to('cuda')
            question_prompt = sample['question_prompt'].to('cuda')
            target = sample['answer_label'].to('cuda')
            question_id = sample['question_id']

            preds_qa, _, _ = model(audio_feat, visual_feat, visual_patch_feat, question_feat, question_len,
                                   question_prompt)
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)
            total += preds.size(0)
            correct += (predicted == target).sum().item()

            question_id = question_id.numpy().tolist()

            for index, ques_id in enumerate(question_id):
                x = questionid_to_samples[ques_id]
                type = ast.literal_eval(x['type'])

                if type[0] == 'Audio':
                    if type[1] == 'Counting':
                        A_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        A_cmp.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Visual':
                    if type[1] == 'Counting':
                        V_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        V_loc.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Audio-Visual':
                    if type[1] == 'Existential':
                        AV_ext.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Counting':
                        AV_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        AV_loc.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        AV_cmp.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Temporal':
                        AV_temp.append((predicted[index] == target[index]).sum().item())

    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count) / len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))
    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp)
                   + sum(AV_cmp)) / (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total


def main():
    args = parser.parse_args()
    print(format("main.py path", '<25'), Path(__file__).resolve())
    for arg in vars(args):
        print(format(arg, '<25'), format(str(getattr(args, arg)), '<'))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build Model
    model = TSAV(
        args=args,
        audio_input_dim=args.audio_input_dim,
        visual_input_dim=args.visual_input_dim,
        qst_input_dim=args.qst_input_dim,
        qst_prompt_input_dim=args.qst_prompt_input_dim,
        hidden_size=args.hidden_size,
        answer_vocab_size=42,
        dropout_p1=args.dropout_p1,
        dropout_p2=args.dropout_p2,
        cx_encoder_layers_num=args.cx_encoder_layers_num,
        cx_nhead=args.cx_nhead,
        cx_d_model=args.cx_d_model,
        cx_dim_feedforward=args.cx_dim_feedforward,
        top_k=args.top_k,
        thread=args.thread,
    )
    model = model.to('cuda')
    test_dataset = AVQA_dataset(label=args.label_test,
                                audios_feat_dir=args.audio_feat_dir, visual_feat_dir=args.visual_feat_dir,
                                qst_prompt_dir=args.qst_prompt_dir, qst_feat_dir=args.qst_feat_dir,
                                transform=transforms.Compose([ToTensor()]), mode_flag='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6,
                             pin_memory=True)

    checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint_file, 'model_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda')
    best_epoch = checkpoint['epoch']
    best_acc = checkpoint['Acc']
    print("-------- checkpoint loading successfully ----------")
    print('********************************************************')
    print('The best epoch ---------- {0}'.format(best_epoch))
    print('The best train acc ------ {0}'.format(best_acc))
    print('********************************************************')
    _ = model_test(model, test_loader, args.label_test)


if __name__ == '__main__':
    main()
