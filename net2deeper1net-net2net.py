if __name__ == "__main__":
    import time
    time_start = time.time()

    #######################################################################################
    # 1. Loading and normalizing CIFAR10
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor()])

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
            self.conv1 = nn.Conv2d(3, 64, 5, padding=2) # 32 -> 32
            self.conv2 = nn.Conv2d(64, 64, 5, padding=2) # 16 -> 16
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 8 -> 8
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1) # 8 -> 8
            self.conv5 = nn.Conv2d(128, 128, 3, padding=1) # 8 -> 8
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(8192, 384)
            self.fc2 = nn.Linear(384, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x)) ## 32x32x3 -> 32x32x64
            x = self.pool(x) ## 32x32x64 -> 16x16x64
            x = F.relu(self.conv2(x)) ## 16x16x64 -> 16x16x64
            x = self.pool(x) ## 16x16x64 -> 8x8x64
            x = F.relu(self.conv3(x)) ## 8x8x64 -> 8x8x128
            x = F.relu(self.conv4(x)) ## 8x8x128 -> 8x8x128
            x = F.relu(self.conv5(x)) ## 8x8x128 -> 8x8x128
            x = x.view(-1, 8*8*128) ## 8x8x128 -> 8192
            x = F.relu(self.fc1(x)) ## 8192 -> 384
            x = F.dropout(x, training=self.training)
            x = self.fc2(x) ## 384 -> 10

            return x

    class Net2Deeper1Net(nn.Module):
        def __init__(self):
            super(Net2Deeper1Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 5, padding=2) # 32 -> 32
            self.conv2 = nn.Conv2d(64, 64, 5, padding=2) # 16 -> 16
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 8 -> 8
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1) # 8 -> 8
            self.conv5 = nn.Conv2d(128, 128, 3, padding=1) # 8 -> 8
            self.conv6 = nn.Conv2d(128, 128, 3, padding=1) # 8 -> 8
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(8192, 384)
            self.fc2 = nn.Linear(384, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x)) ## 32x32x3 -> 32x32x64
            x = self.pool(x) ## 32x32x64 -> 16x16x64
            x = F.relu(self.conv2(x)) ## 16x16x64 -> 16x16x64
            x = self.pool(x) ## 16x16x64 -> 8x8x64
            x = F.relu(self.conv3(x)) ## 8x8x64 -> 8x8x128
            x = F.relu(self.conv4(x)) ## 8x8x128 -> 8x8x128
            x = F.relu(self.conv5(x)) ## 8x8x128 -> 8x8x128
            x = F.relu(self.conv6(x)) ## 8x8x128 -> 8x8x128
            x = x.view(-1, 8*8*128) ## 8x8x128 -> 8192
            x = F.relu(self.fc1(x)) ## 8192 -> 384
            x = F.dropout(x, training=self.training)
            x = self.fc2(x) ## 384 -> 10

            return x

    net = Net2Deeper1Net()
    net_teacher = Net()

    #######################################################################################

    net_teacher.load_state_dict(torch.load('./checkpoint/checkpoint_teacher_399.pth'))

    net._modules['conv1'].weight.data = net_teacher._modules['conv1'].weight.data
    net._modules['conv2'].weight.data = net_teacher._modules['conv2'].weight.data
    net._modules['conv3'].weight.data = net_teacher._modules['conv3'].weight.data
    net._modules['conv4'].weight.data = net_teacher._modules['conv4'].weight.data
    net._modules['conv5'].weight.data = net_teacher._modules['conv5'].weight.data
    net._modules['conv6'].weight.data = net_teacher._modules['conv5'].weight.data

    #######################################################################################
    ## 3. Define a Loss function and optimizer≠
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #######################################################################################
    ## 4. Train the network
    epoch_max = 1000

    train_loss_list = []
    test_loss_list = []
    accuracy_list = []

    for epoch in range(epoch_max):
        running_loss = 0.0
        test_loss = 0.0

        train_num = 0
        test_num = 0

        #### Checking train loss
        time_start_loop = time.time()
        for i, data in enumerate(trainloader, 0):
            if i % 50 == 0:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_num = train_num + 1
        train_loss_list.append(running_loss)
        print('iteration: %d ==> loss: %.3f' % (epoch + 1, running_loss / train_num))
        print('Required time(s): ' + str(time.time() - time_start_loop))

        #### Checking test loss
        time_start_loop = time.time()
        for i, data in enumerate(testloader, 0):
            if i % 50 == 0:
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_num = test_num + 1
        test_loss_list.append(test_loss)
        print('iteration: %d ==> test_loss: %.3f' % (epoch + 1, test_loss / test_num))
        print('Required time(s): ' + str(time.time() - time_start_loop))

        ## Checking accuracy
        time_start_test = time.time()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                if i % 50 == 0:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
        print('Required time(s): ' + str(time.time() - time_start_test))

        time_start_save = time.time()
        filename = './checkpoint/checkpoint_deeper1_net2net_' + str(epoch) + '.pth'
        torch.save(net.state_dict(), filename)
        print('save complete!')

    print('Finished Training')

    #######################################################################################
    ## 5. Test the network on the test data
    time_start_test = time.time()

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
    print('Required time(s): ' + str(time.time() - time_start_test))
    print('Total required time(s): ' + str(time.time() - time_start))
