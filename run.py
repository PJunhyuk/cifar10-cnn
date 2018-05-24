if __name__ == "__main__":
    #######################################################################################
    # 1. Loading and normalizing CIFAR10
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/tmp/data/cifar10-py', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/tmp/data/cifar10-py', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truch')

    #######################################################################################
    # 2. Define a Convolution Neural Network

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 5)
            self.conv2 = nn.Conv2d(64, 128, 5)
            self.conv3 = nn.Conv2d(128, 128, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(3 * 3 * 128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x)) ## 32x32x3 -> 28x28x64
            x = self.pool(x) ## 28x28x64 -> 14x14x64
            x = F.relu(self.conv2(x)) ## 14x14x64 -> 10x10x128
            x = self.pool(x) ## 10x10x128 -> 5x5x128
            x = F.relu(self.conv3(x)) ## 5x5x128 -> 3x3x128
            x = x.view(-1, 3 * 3 * 128) ## 3x3x128 -> 1152
            x = self.fc(x) ## 1152 -> 10
            return x

    net = Net()

    #######################################################################################
    ## 3. Define a Loss function and optimizer
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #######################################################################################
    ## 4. Train the network
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    #######################################################################################
    ## 5. Test the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
