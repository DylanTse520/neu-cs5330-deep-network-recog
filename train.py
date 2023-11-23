# Yixiang Xie
# CS 5330 F23
# Build and train a CNN network to recognize MNIST digits


# import statements
import os
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# define the convolutional neural network
class Net(nn.Module):
    # function to initialize the layers of the network
    def __init__(self):
        super(Net, self).__init__()
        # conv1 is a convolutional layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # conv2 is a convolutional layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # this is a dropout layer with a probability of 0.5
        self.dropout = nn.Dropout2d()
        # fc1 is a fully connected layer with 50 neurons
        self.fc1 = nn.Linear(320, 50)
        # fc2 is a fully connected layer with 10 neurons
        self.fc2 = nn.Linear(50, 10)

    # function to compute a forward pass for the network
    def forward(self, x):
        # x is 1x28x28. conv1 is applied here
        x = self.conv1(x)
        # x is 10x24x24
        # a max pooling layer with a 2x2 window and a ReLU function are applied here
        x = F.relu(F.max_pool2d(x, 2))
        # x is 10x12x12. conv2 is applied here
        x = self.conv2(x)
        # x is 20x8x8. a dropout layer is applied here
        x = self.dropout(x)
        # x is 20x8x8
        # a max pooling layer with a 2x2 window and a ReLU function are applied here
        x = F.relu(F.max_pool2d(x, 2))
        # x is 20x4x4. x is flattened here
        x = x.view(-1, 320)
        # x is 320x1. fc1 and a ReLU function are applied here
        x = F.relu(self.fc1(x))
        # x is 50x1. fc2 and a log softmax function are applied here
        x = F.log_softmax(self.fc2(x), dim=1)
        # x is 10x1
        return x


# function to train the network
def train_network(
    network: Net,
    optimizer: optim.SGD,
    epoch: int,
    train_loader: DataLoader,
    train_losses: list,
    train_counter: list,
    batch_size: int,
    save_path: str,
):
    # set the network to training mode
    network.train()
    # for each batch in the training data
    for batch_idx, (data, target) in enumerate(train_loader):
        # set the gradients to zero
        optimizer.zero_grad()
        # compute the output of the network
        output = network(data)
        # compute the loss
        loss = F.nll_loss(output, target)
        # compute the gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # for every 10 batches
        if batch_idx % 10 == 0:
            # print the loss
            print(
                "Train epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            # save the loss and counter
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset))
            )
            # create ./results if it does not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # save the model and optimizer
            torch.save(network.state_dict(), save_path + "/model.pth")
            torch.save(optimizer.state_dict(), save_path + "/optimizer.pth")


# function to test the network
def test_network(
    network: Net,
    test_loader: DataLoader,
    test_losses: list,
):
    # set the network to evaluation mode
    network.eval()
    # initialize the test loss and correct counter
    test_loss = 0
    correct = 0
    # for each batch in the testing data (1000 images per batch)
    with torch.no_grad():
        for data, target in test_loader:
            # compute the output of the network
            output = network(data)
            # compute the loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # increment the correct counter if the prediction is correct
            correct += pred.eq(target.data.view_as(pred)).sum()
    # compute the average test loss
    test_loss /= len(test_loader.dataset)
    # append the test loss to the list
    test_losses.append(test_loss)
    # print the test loss and accuracy
    print(
        "\nTest average loss: {:.4f}\t\tAccuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


# main function
def main(argv):
    # import MNIST train data
    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    # create a data loader for the train data
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # show the first six example digits
    for i in range(6):
        # plot each image in the grid
        plt.subplot(2, 3, i + 1)
        # plot the image in grayscale
        plt.imshow(train_data.data[i], cmap="gray", interpolation="none")
        # plot the label as the title of the image
        plt.title("Label {}".format(train_data.targets[i]))
        # do not show the axes
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # import MNIST test data
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    # create a data loader for the test data
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)

    # initialize the network
    network = Net()
    # initialize the optimizer
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    # train the network for 5 epochs
    n_epochs = 5
    # initialize the training and testing losses and counters
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    # test the network before training
    test_network(network, test_loader, test_losses)
    # train the network for 5 epochs, and test it after each epoch
    for epoch in range(1, n_epochs + 1):
        train_network(
            network,
            optimizer,
            epoch,
            train_loader,
            train_losses,
            train_counter,
            64,
            "./results/MNIST",
        )
        test_network(network, test_loader, test_losses)

    # plot the training loss
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
