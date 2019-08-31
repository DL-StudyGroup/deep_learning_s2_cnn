import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
# import sys
# import os
#
# # sys.path.append(os.path.curdir)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                padding=2,  # if stride=1, padding=(kernel_size - 1) /2 = (5 -1)/2
            ),  # -> (16, 28, 28)
            nn.ReLU(),  # -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),  # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(2)  # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output


def test():
    test_data = torchvision.datasets.MNIST(root='../mnist/', train=False)
    # shape from (2000, 28,28) to (2000, 1, 28,28), value in range(0, 1)
    test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[0] / 255.
    test_x1 = Variable(torch.unsqueeze(test_data.data, dim=1))[0]
    print("test_x1:", test_x1)
    print("test_x1.shape:", test_x1.shape)


def get_test_x():
    test_data = torchvision.datasets.MNIST(root='../mnist/', train=False)
    # shape from (2000, 28,28) to (2000, 1, 28,28), value in range(0, 1)
    var = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:10] / 255.
    return var


def train_net():
    # Hyper Parameters
    EPOCH = 1
    BATCH_SIZE = 50
    LR = 0.001
    DOWNLOAD_MNIST = False
    train_data = torchvision.datasets.MNIST(
        root='../mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),  # (0, 1)   (0-255)
        download=DOWNLOAD_MNIST
    )

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.MNIST(root='../mnist/', train=False)
    # shape from (2000, 28,28) to (2000, 1, 28,28), value in range(0, 1)
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255.
    test_y = test_data.test_labels[:2000]

    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            # b_x = Variable(x)
            # b_y =
            output = cnn(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / test_y.size(0)
                # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum())/ float(test_y.size(0))
                # print('training pred_y', pred_y, ' sum of matches: ',
                #       float((pred_y == test_y.data.numpy()).astype(int).sum()))
                # print('training test_y', test_y, 'test_y.size(0)', test_y.size(0))

                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    torch.save(cnn, "cnn.pkl")
    torch.save(cnn.state_dict(), "cnn_params.pkl")
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')
