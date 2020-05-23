
import os
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-init', type=str,
                        default="")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    model = ResNet18Backbone(pretrained=False).cuda()
    model.load_state_dict(torch.load('epoch_1.pth', map_location=torch.device('cuda')))
    # raise NotImplementedError("TODO: build model and load weights snapshot")

    # dataset
    data_root = "/home/nallapar/workspace/ex1/crops"
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True)
    # raise NotImplementedError("Load the validation dataset (crops), use the transform above.")

    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [25, 49, 88, 103]
    nns = []
    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img, val_loader, 5, idx)
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(img[0].T.numpy())
        axes[1].imshow(val_loader.dataset[closest_idx][0].T.numpy())
        plt.savefig(f"orig_and_nn_{idx}.jpg")
        #raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")


def find_nn(model, query_img, loader, k, idx):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    with torch.no_grad():
        query_img = query_img.to('cuda')
        query_img_pred = model(query_img)
        dists = []
        for i, x in enumerate(loader):
            if i == idx:
                continue
            x = x.to('cuda')
            pred = model(x)
            dist = np.linalg.norm(pred.cpu() - query_img_pred.cpu())
            dists.append(dist)
        dists = np.array(dists)
        inds = dists.argsort()[:k]
        vals = dists[inds]
        return inds[0], vals[0]
        # raise NotImplementedError("TODO: nearest neighbors retrieval")
        # return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
