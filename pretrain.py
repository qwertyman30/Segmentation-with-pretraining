import os
import numpy as np
import argparse
import torch
from pprint import pprint
from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from utils import check_dir, accuracy, get_logger
from models.pretraining_backbone import ResNet18Backbone
from torch import nn
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--weights-init', type=str,
                        default="random")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--bs', type=int, default=8, help='batch_size')
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)

    # build model and load weights
    model = ResNet18Backbone(pretrained=False).cuda()
    model.load_state_dict(torch.load('pretrain_weights_init.pth', map_location=torch.device('cuda'))['model'])
    #model = torch.load('pretrain_weights_init.pth')#, map_location=torch.device('cpu'))

    # load dataset
    data_root = args.data_folder
    train_transform, val_transform = get_transforms_pretraining(args)
    train_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "train"), transform=train_transform)
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)

    # TODO: loss function
    criterion = nn.CrossEntropyLoss().cuda()
#     raise NotImplementedError("TODO: loss function")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_model = None
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    losses, val_losses, val_accs = [], [], []
    for epoch in range(10):
        print("Epoch {}".format(epoch))
        loss = train(train_loader, model, criterion, optimizer)
        losses.append(loss)
        val_loss, val_acc = validate(val_loader, model, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        torch.save(model.state_dict(), f'epoch_{epoch}.pth')

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
#             raise NotImplementedError("TODO: save model if a new best validation error was reached")
    _, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].plot(range(1, 11), losses)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train")

    axes[1].plot(range(1, 11), val_losses)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation losses")

    axes[2].plot(range(1, 11), val_accs)
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Validation accuracies")
    plt.savefig("Results.png")


# train one epoch over the whole training dataset. You can change the method's signature.
def train(loader, model, criterion, optimizer):
    losses = []
    for X_train, y_train in loader:
        X_train = X_train.to('cuda', non_blocking=True)
        y_train = y_train.to('cuda', non_blocking=True)
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.item())
    return np.mean(losses)

# validation function. you can change the method's signature.
def validate(loader, model, criterion):
    losses, accuracies = [], []
    with torch.no_grad():
        for X_val, y_val in loader:
            X_val = X_val.to('cuda', non_blocking=True)
            y_val = y_val.to('cuda', non_blocking=True)
            y_preds = model(X_val)
            loss = criterion(y_preds, y_val)
            _, y_preds = y_preds.max(1)
            accuracy = 100.0 * sum(y_val == y_preds)/len(y_preds)
            losses.append(loss.item())
            accuracies.append(accuracy.item())
        print(loss.item(), accuracy.item())
    return np.mean(losses), np.mean(accuracies)


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
