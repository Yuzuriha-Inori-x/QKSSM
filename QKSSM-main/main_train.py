import sys

sys.path.append("../TSAV-main")
import warnings
import random
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataloader import *
from nets.TSAV import TSAV
from configs.arguments import parser

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')

print("\n--------------- TSAV --------------- \n")


def model_train(args, model, train_loader, optimizer, criterion, epoch=0):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):
        audio_feat = sample['audio_feat'].to('cuda')
        visual_feat = sample['visual_feat'].to('cuda')
        visual_patch_feat = sample['visual_patch_feat'].to('cuda')
        question_feat = sample['question_feat'].to('cuda')
        question_len = sample['question_len'].to('cuda')
        question_prompt = sample['question_prompt'].to('cuda')
        target = sample['answer_label'].to('cuda')

        optimizer.zero_grad()
        out_qa, CL_loss = model(audio_feat, visual_feat, visual_patch_feat, question_feat, question_len,
                                question_prompt)

        loss_qa = criterion(out_qa, target)
        CL_loss = args.loss_nce_wei * CL_loss
        loss = loss_qa + CL_loss

        loss.backward()
        optimizer.step()

        pred_index, predicted = torch.max(out_qa, 1)
        correct_qa += (predicted == target).sum().item()
        total_qa += out_qa.size(0)

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t total Loss: {:.4f} |  CE_loss:{:.6f}  CL-loss:{:.4f}'.format(
                    epoch, batch_idx * len(audio_feat), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), loss_qa.item(),
                    CL_loss.item()), flush=True)

    return correct_qa, total_qa, 100 * correct_qa / total_qa


def model_eval(model, val_loader):
    model.eval()
    total_qa = 0
    correct_qa = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio_feat = sample['audio_feat'].to('cuda')
            visual_feat = sample['visual_feat'].to('cuda')
            visual_patch_feat = sample['visual_patch_feat'].to('cuda')
            question_feat = sample['question_feat'].to('cuda')
            question_len = sample['question_len'].to('cuda')
            question_prompt = sample['question_prompt'].to('cuda')
            target = sample['answer_label'].to('cuda')

            preds_qa, _ = model(audio_feat, visual_feat, visual_patch_feat, question_feat, question_len,
                                question_prompt)
            _, predicted = torch.max(preds_qa, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Accuracy val_set qa: %.2f %%' % (100 * correct_qa / total_qa), flush=True)
    return 100 * correct_qa / total_qa


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
    train_dataset = AVQA_dataset(label=args.label_train,
                                 audios_feat_dir=args.audio_feat_dir, visual_feat_dir=args.visual_feat_dir,
                                 qst_prompt_dir=args.qst_prompt_dir, qst_feat_dir=args.qst_feat_dir,
                                 transform=transforms.Compose([ToTensor()]), mode_flag='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6,
                              pin_memory=True)

    val_dataset = AVQA_dataset(label=args.label_test,
                               audios_feat_dir=args.audio_feat_dir, visual_feat_dir=args.visual_feat_dir,
                               qst_prompt_dir=args.qst_prompt_dir, qst_feat_dir=args.qst_feat_dir,
                               transform=transforms.Compose([ToTensor()]), mode_flag='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr_step, gamma=args.steplr_gamma)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nthe {epoch}-th learning rate is {optimizer.param_groups[0]['lr']}")
        #########################################################################################################
        # !!! train
        correct_qa, total_qa, train_acc = model_train(args, model, train_loader, optimizer, criterion, epoch=epoch)
        print('Accuracy train_set qa: %.2f %%' % (100 * correct_qa / total_qa), flush=True)
        scheduler.step(epoch)

        #########################################################################################################
        # !!! val
        current_acc = model_eval(model, val_loader)
        if current_acc >= best_acc:
            model_to_save = model.module if hasattr(model, 'module') else model
            optimizer_to_save = optimizer
            best_acc = current_acc
            save_model_folder = Path(args.model_save_dir, args.checkpoint_file)
            # save_epoch_model_path = Path(save_model_folder, f"model_{epoch}.pt")
            save_best_model_path = Path(save_model_folder, f"model_best.pt")
            if not os.path.isdir(save_model_folder):
                os.mkdir(save_model_folder)
            # epoch_save_dict = {
            #     'epoch': epoch,
            #     'model_state_dict': model_to_save.state_dict(),
            #     'optimizer_state_dict': optimizer_to_save.state_dict(),
            #     'Acc': best_acc,
            # }
            # torch.save(epoch_save_dict, save_epoch_model_path)
            best_save_dict = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer_to_save.state_dict(),
                'Acc': best_acc,
            }
            torch.save(best_save_dict, save_best_model_path)
            print("Best Acc: %.2f %%" % best_acc)
            print("Best Epoch: ", best_epoch)
            print("*" * 20)


if __name__ == '__main__':
    main()
