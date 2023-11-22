# Yixiang Xie
# CS 5330 F23
# Test a CNN network for recognizing MNIST digits


# import statements
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from train import Net


# function to test the network on first 10 testing images
def test_network(
    network: nn.Module,
    test_loader: DataLoader,
):
    # set the network to evaluation mode
    network.eval()
    # store the predicted labels
    predicted_labels = []

    # for each image in the testing data
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            # compute the output of the network
            output = network(data)
            # print the output, rounded to 2 decimal places
            print(
                "Image {} output: {}".format(
                    idx + 1, [round(o.item(), 2) for o in output[0]]
                )
            )
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # print the prediction
            print("Image {} prediction: {}".format(idx + 1, pred[0][0]))
            predicted_labels.append(pred[0][0])
            # print the label
            print("Image {} label: {}".format(idx + 1, target.item()))

            # plot only the first nine images
            if idx == 9:
                continue
            # plot each image in the grid
            plt.subplot(3, 3, idx + 1)
            # plot the image in grayscale
            plt.imshow(data.squeeze().numpy(), cmap="gray", interpolation="none")
            # plot the label as the title of the image
            plt.title("Prediction {}".format(predicted_labels[idx]), y=0.95)
            # do not show the axes
            plt.xticks([])
            plt.yticks([])

    plt.show()


# main function
def main(argv):
    # import MNIST testing data
    testing_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    # create a subset of the testing data with the first 10 images
    subset_testing_data = Subset(testing_data, range(10))
    # create a data loader for the testing data
    test_loader = DataLoader(subset_testing_data, batch_size=1)

    # read the model from file
    network = Net()
    network.load_state_dict(torch.load("./results/model.pth"))

    # test the network on the first 10 images
    test_network(network, test_loader)

    return


if __name__ == "__main__":
    main(sys.argv)
