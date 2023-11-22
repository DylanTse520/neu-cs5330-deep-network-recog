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


# function to test the network on first 10 MNIST testing images
def test_network_MNIST(
    network: nn.Module,
    test_loader: DataLoader,
):
    # set the network to evaluation mode
    network.eval()
    # store the predicted labels
    predicted_labels = []

    # create a new figure
    plt.figure()

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


# function to test the network on 10 handwritten digit images
def test_network_handwritten(
    network: nn.Module,
    test_loader: DataLoader,
):
    # set the network to evaluation mode
    network.eval()
    # initialize the test loss and correct counter
    test_loss = 0
    correct = 0

    # create a new figure
    plt.figure()

    # for each image in the testing data
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            # compute the output of the network
            output = network(data)
            # compute the loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # increment the correct counter if the prediction is correct
            correct += pred.eq(target.data.view_as(pred)).sum()

            # plot each image in the grid
            plt.subplot(2, 5, idx + 1)
            # plot the image in grayscale
            plt.imshow(data.squeeze().numpy(), cmap="gray", interpolation="none")
            # plot the label as the title of the image
            plt.title("Prediction {}".format(pred[0][0]))
            # do not show the axes
            plt.xticks([])
            plt.yticks([])

    # compute the average test loss
    test_loss /= len(test_loader.dataset)
    # print the test loss and accuracy
    print(
        "\nTest average loss: {:.4f}\t\tAccuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


# handwritten digits set transform
class HandwrittenTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        return transforms.functional.invert(x)


# main function
def main(argv):
    # read the model from file
    network = Net()
    network.load_state_dict(torch.load("./results/model.pth"))

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

    # test the network on the MNIST subset
    test_network_MNIST(network, test_loader)

    # import the 10 handwritten digit images
    handwritten_data = datasets.ImageFolder(
        root="./data/handwritten",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                HandwrittenTransform(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    # create a data loader for the handwritten data
    handwritten_loader = DataLoader(handwritten_data, batch_size=1)

    # test the network on the handwritten images
    test_network_handwritten(network, handwritten_loader)

    plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
