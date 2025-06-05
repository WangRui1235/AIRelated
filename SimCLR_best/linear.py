import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils
from utils import Model


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(out.data, 1)
            total_correct += (predicted == target).sum().item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC: {:.2f}% '
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='0.5_512_300_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', default=512, type=int, help='批量大小')
    parser.add_argument('--epochs', default=500, type=int, help='训练轮数')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    train_data = CIFAR10(root='../data', train=True, transform=utils.train_transform, download=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = CIFAR10(root='../data', train=False, transform=utils.test_transform, download=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    stop = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_val(model, train_loader, optimizer)
        test_loss, test_acc = train_val(model, test_loader, None)
        if test_acc > best_acc:
            best_acc = test_acc
            stop = 0
            torch.save(model.state_dict(), 'linear_model.pth')
        stop += 1 
        if stop > 6 or epoch == epochs:
            break
