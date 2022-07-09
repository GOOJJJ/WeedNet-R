import argparse
import collections
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from retinanet import utils
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv' ,help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train',default='./dataset/train.csv', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes',default='./dataset/class_list.csv', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default='./dataset/val.csv',help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--backbone',default='resnet',help='select backbone form resnet ,resnext')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=20)
    parser.add_argument('--batch_size', help='batch size', type=int, default=2)
    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.backbone=='resnet':
        if parser.depth == 18:
            retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 34:
            retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 152:
            retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    elif parser.backbone=='resnext':
        if parser.depth == 50:
            retinanet = model.resnext50(num_classes=dataset_train.num_classes(),phi=0,pretrained=True)
        elif parser.depth == 152:
            retinanet = model.resnext101(num_classes=dataset_train.num_classes(),phi=1 ,pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True


    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True
    print(retinanet.module)

    import pytorch_warmup as warmup
    optimizer = optim.Adam(retinanet.parameters(),lr=1e-4)
    num_steps =len(dataloader_train)*parser.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_steps,eta_min=1e-8)
    warmup_scheduler=warmup.UntunedExponentialWarmup(optimizer)

    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    save_base_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(save_base_dir):
        os.mkdir(save_base_dir)

    import datetime
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    save_dir=os.path.join(save_base_dir,time_str)
    save_plots_dir=os.path.join(save_dir,'plots')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_plots_dir):
        os.mkdir(save_plots_dir)

    train_info = 'baseline:SGD,scheduler.LRstep(5,0.1),bs=4,epoch=25,init_lr=1e-2,SSH*5'

    with open(os.path.join(save_dir, "train info" + ".txt"), 'w') as f:
        f.write(train_info + '\n')
        f.write('batch size: %d \n'%parser.batch_size)
        f.write('epoch: %d \n'%parser.epochs)
        f.write('Num training images: {}\n'.format(len(dataset_train)))
        f.write('Num valid images: {}\n'.format(len(dataset_val)))
        f.write('Model architecture:\n %s\n' % (retinanet.module))

    train_loss=[]
    mAPs={'mAP':[]}
    class_map={0:'weed',1:'crop'}

    for label in range(dataset_val.num_classes()):
        mAPs[label]=[]
    lrs=[]
    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                warmup_scheduler.dampen()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | learning rate: {:1.8f} |Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, get_lr(optimizer),float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        lrs.append(get_lr(optimizer))
        with open(os.path.join(save_plots_dir, "learn_rate" + ".txt"), 'a') as f:
            f.write(str(get_lr(optimizer)))
            f.write("\n")
        utils.lr_plot(lrs,save_plots_dir)

        train_loss.append(np.mean(epoch_loss))
        with open(os.path.join(save_plots_dir, "epoch_loss"  + ".txt"), 'a') as f:
            f.write(str(np.mean(epoch_loss)))
            f.write("\n")

        utils.loss_plot(train_loss,save_plots_dir)

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP ,_,_ = csv_eval.evaluate(dataset_val, retinanet,save_path=save_plots_dir)
            map=0
            for key in mAP.keys():
                 mAPs[key].append(mAP[key][0])
                 map+=mAP[key][0]
                 with open(os.path.join(save_plots_dir, "%s_AP"%class_map[key] + ".txt"), 'a') as f:
                    f.write(str(mAP[key][0]))
                    f.write("\n")

            mAPs['mAP'].append(map/dataset_val.num_classes())

            with open(os.path.join(save_plots_dir, "mAP" + ".txt"), 'a') as f:
                f.write(str(map/dataset_val.num_classes()))
                f.write("\n")

            utils.mAP_plot(mAPs,class_map,save_plots_dir)

        torch.save(retinanet.module, os.path.join(save_dir,'{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)))

    retinanet.eval()
    torch.save(retinanet, os.path.join(save_dir,'model_final.pt'))


if __name__ == '__main__':
    main()
