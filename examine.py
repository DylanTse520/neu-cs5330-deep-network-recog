# Yixiang Xie
# CS 5330 F23
# Examine a CNN network for recognizing MNIST digits


# import statements
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from train import Net
import cv2
import numpy as np


# function to analyze and visualize the first layer of the network
def analyze_first_layer(
    network: Net,
):
    # access the first layer of the network
    first_layer = network.conv1
    # get the weights of the first layer
    weights = first_layer.weight.data

    # create a figure for the weights
    plt.figure()

    # for each filter
    for i in range(weights.shape[0]):
        # print the weights and the shape of the weights
        print("Filter {} weights: {}".format(i + 1, weights[i, 0]))
        print("Filter {} weights shape: {}".format(i + 1, weights[i, 0].shape))
        # plot the weights and the title
        plt.subplot(3, 4, i + 1)
        plt.imshow(weights[i, 0], cmap="gray")
        plt.title("Filter {}".format(i + 1), y=0.95)
        # do not show the axes
        plt.xticks([])
        plt.yticks([])


# function to apply the first layer 10 filters of the network to an image
def apply_filters(
    network: Net,
    image: torch.Tensor,
):
    # access the first layer of the network
    first_layer = network.conv1
    # get the weights of the first layer
    weights = first_layer.weight.data

    # convert the image to numpy array
    image_np = image.numpy().squeeze()
    # normalize and convert to suitable format for OpenCV
    image_np = (
        (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np)) * 255
    )
    image_np = image_np.astype(np.uint8)
    # invert the image
    image_np = 255 - image_np

    # create a figure for the visualizations
    plt.figure()

    # for each filter
    for i in range(weights.shape[0]):
        # convert the filter to numpy array
        filter_np = weights[i, 0].numpy().squeeze()

        # apply filter using OpenCV's filter2D function
        filtered_image = cv2.filter2D(image_np, -1, filter_np)

        # plot the weights of the filter
        plt.subplot(5, 4, 2 * i + 1)
        plt.imshow(weights[i, 0], cmap="gray")
        # do not show the axes
        plt.xticks([])
        plt.yticks([])
        # plot the filtered image
        plt.subplot(5, 4, 2 * i + 2)
        plt.imshow(filtered_image, cmap="gray")
        # do not show the axes
        plt.xticks([])
        plt.yticks([])


# main function
def main(argv):
    # read the model from file
    network = Net()
    network.load_state_dict(torch.load("./results/model.pth"))

    # print the model
    print(network)

    # analyze the first layer
    analyze_first_layer(network)

    # import MNIST train data
    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    # get the first image from the train data
    train_image = train_data[0][0]

    # apply the first layer 10 filters to the image
    apply_filters(network, train_image)
    
    plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
