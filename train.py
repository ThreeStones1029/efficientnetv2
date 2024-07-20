import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate
from tools.data.dataset_process import read_split_dataset, read_from_split_folder


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    if os.path.exists(args.model_save_dir) is False:
        os.makedirs(args.model_save_dir)

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_dataset(args.data_path, 
    #                                                                                               split_ratio={"train": 0.7, "val": 0.3}, 
    #                                                                                               resplit=True, 
    #                                                                                               save_txt=True)
    train_images_path, train_images_label, val_images_path, val_images_label = read_from_split_folder(args.data_path)
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    
    num_model = args.weights_category

    data_transform = {"train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                      "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                            transforms.CenterCrop(img_size[num_model][1]),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 根据类型导入[s, m, l]初始化模型
    if args.weights_category == "s":
        model = efficientnetv2_s(num_classes=args.num_classes).to(device)
    if args.weights_category == "m":
        model = efficientnetv2_m(num_classes=args.num_classes).to(device)
    if args.weights_category == "l":
        model = efficientnetv2_l(num_classes=args.num_classes).to(device)
    
    if args.pretrain_weights != "":
        if os.path.exists(args.pretrain_weights):
            weights_dict = torch.load(args.pretrain_weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.pretrain_weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    val_best_model_acc = 0
    train_best_model_acc = 0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        # validate
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if args.only_save_best_model:
            if val_acc > val_best_model_acc:
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, "val_best_model.pth"))
                val_best_model_acc = val_acc
            # if train_acc > train_best_model_acc:
            #     torch.save(model.state_dict(), os.path.join(args.model_save_dir, "train_best_model.pth"))
            #     train_best_model_acc = train_acc
        else:
            if epoch % args.snapshot_epoch == 0:
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, "model-{}.pth".format(epoch)))
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, "final_model.pth"))

    print("[*] The val best model acc is {}".format(val_best_model_acc))
    print("[*] The train best model acc is {}".format(train_best_model_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--snapshot_epoch', type=int, default=5)

    parser.add_argument('--only_save_best_model', type=bool, default=True)
    # dataset path
    parser.add_argument('--data-path', type=str, default="dataset/TD20240705_LA/split_dataset/fold5/cut_dataset")
    # download model pre_weights
    parser.add_argument('--pretrain_weights', type=str, default='weights/LA_preoperative_xray_fracture_cut_complete/m/val_best_model.pth', help='pretrain weights path')
    parser.add_argument("--weights_category", type=str, default="m", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_save_dir', type=str, default="weights/TD20240705_LA/preoperative_pretrain/fold5/m", help="trained models save path")
    parser.add_argument('--log_dir', type=str, default="runs/TD20240705_LA/preoperative_pretrain/fold5/m", help="tensorboard logdir save path")
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
