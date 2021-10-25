from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import visdom
# vis = visdom.Visdom(env='adaptive_lr')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def lr_adap(lr, loss_diff, norm):
    new_lr = (lr/2 * norm**2 + loss_diff) / (norm**2 + 1e-6)
    return new_lr.data.item()

losses = [0, 0]

def train(args, model, device, optimizer, train_loader, epoch, lr, train_loss, lr_list, loss_diff_list):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        losses[batch_idx % 2] = loss
        loss_diff = torch.max(torch.tensor([losses[batch_idx % 2] - losses[(batch_idx+1) % 2],0]))
        # print(grad_norm(model.parameters()), lr, loss_diff)
        lr = lr_adap(lr, loss_diff, grad_norm(model.parameters()))
        lr_list.append(lr)
        loss_diff_list.append(loss_diff.item())
        # print(lr.data.item())
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # vis.line([loss.item()], [(epoch - 1)*len(train_loader) + batch_idx],
            #          win='train_loss_compenlr',
            #          opts={'title':'compen','legend':['loss']},
            #          update= None if (epoch - 1)*len(train_loader) + batch_idx == 0 else 'append')
            train_loss.append(loss.item())
            if args.dry_run:
                break
    return lr, train_loss, lr_list, loss_diff_list


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # vis.line([test_loss], [epoch],
    #          win='test_loss_compenlr',
    #          opts={'title': 'compen', 'legend': ['test_loss']},
    #          update= 'append')
    return test_loss, 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_loss = []
    test_loss = []
    test_acc = []
    lr_list = []
    loss_diff_list = []
    for epoch in range(1, args.epochs + 1):
        lr, train_loss, lr_list, loss_diff_list = train(args, model, device, optimizer, train_loader, epoch,
                                        lr, train_loss, lr_list, loss_diff_list)
        a, b = test(model, device, test_loader, epoch)
        test_loss.append(a)
        test_acc.append(b)
        # scheduler.step()

    with open('../data/MNIST/results_full.txt', 'a+') as fl:
        fl.write('Optimizer is AdaComp, LR is {}, batch size is {} \n '
                 'train loss is {} \n test loss is{} \n test acc is {} \n lr list is {} \n loss_diff list is {} \n'.
                 format(args.lr, args.batch_size, train_loss, test_loss, test_acc, lr_list, loss_diff_list))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
