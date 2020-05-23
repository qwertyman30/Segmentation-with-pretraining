import os
import pprint
import random
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
import argparse
import torch
from torch import nn
import time
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger
from models.second_segmentation import Segmentator
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderBinarySegmentation

set_random_seed(0)
global_step = 0


set_random_seed(0)
global_step = 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data")
    parser.add_argument('weights_init', type=str, default="ImageNet")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'dt_binseg', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)

    # model
    pretrained_model = ResNet18Backbone(pretrained=False).cuda()
    pretrained_model.load_state_dict(torch.load(args.weights_init, map_location=torch.device('cuda')))
    #raise NotImplementedError("TODO: build model and load pretrained weights")
    model = Segmentator(2, pretrained_model.features, img_size).cuda()

    # dataset
    train_trans, val_trans, train_target_trans, val_target_trans = get_transforms_binary_segmentation(args)
    data_root = args.data_folder
    train_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_trans,
        target_transform=train_target_trans
    )
    val_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_trans,
        target_transform=val_target_trans
    )
    print("Dataset size: {} samples".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=6, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=6, pin_memory=True, drop_last=False)

    # TODO: loss
    criterion = nn.CrossEntropyLoss().cuda()
    # TODO: SGD optimizer (see pretraining)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #raise NotImplementedError("TODO: loss function and SGD optimizer")

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_miou = 0.0
    for epoch in range(100):
        logger.info("Epoch {}".format(epoch))
        t_loss = train(train_loader, model, criterion, optimizer, logger)
        print(f"Train Loss: {t_loss}")
        val_loss, val_iou = validate(val_loader, model, criterion, logger, epoch)
        print(f"Val Loss: {t_loss} Val iou: {val_iou}")
        if best_val_miou < val_iou:
            best_val_miou = val_iou
            save_model(model, optimizer, args, epoch, val_loss, val_iou, logger)
        # TODO save model


def train(loader, model, criterion, optimizer, logger):
    losses = []
    for X_train, y_train in loader:
        X_train = X_train.to('cuda', non_blocking=True)
        y_train = y_train.to('cuda', non_blocking=True)
        _, y_pred = model(X_train).max(1, True)
        #_, y_pred = y_pred.max(0)
        #print(y_pred[0,:,0,0])
        print('y_pred', y_pred.dtype, y_pred.shape)
        print('y_train', y_train.dtype, y_train.shape)
        loss = criterion(y_train, y_pred)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info(f"Loss: {loss.item()}")
    print(loss.item())
    return np.mean(losses)
    #raise NotImplementedError("TODO: training routine")


def validate(loader, model, criterion, logger, epoch=0):
    losses, ious = [], []
    with torch.no_grad():
        for X_val, y_val in loader:
            X_val = X_val.to('cuda', non_blocking=True)
            y_val = y_val.to('cuda', non_blocking=True)
            y_preds = model(X_val)
            loss = criterion(y_preds, y_val)
            _, y_preds = y_preds.max(1)
            iou = mIoU(y_preds, y_val)
            losses.append(loss.item())
            ious.append(iou)
        print(loss.item(), iou)
    return np.mean(losses), np.mean(ious)
    # raise NotImplementedError("TODO: validation routine")
    # return mean_val_loss, mean_val_iou


def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = 'BEST' if best else ''
    logger.info('==> Saving '+add_text_best+' ... epoch{} loss{:.03f} miou{:.03f} '.format(epoch, val_loss, val_iou))
    state = {
        'opt': args,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_loss,
        'miou': val_iou
    }
    if best:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_best.pth'))
    else:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_epoch{}_loss{:.03f}_miou{:.03f}.pth'.format(epoch, val_loss, val_iou)))


if __name__ == '__main__':
    args = parse_arguments()
    #pprint(vars(args))
    print('1')
    print()
    main(args)
