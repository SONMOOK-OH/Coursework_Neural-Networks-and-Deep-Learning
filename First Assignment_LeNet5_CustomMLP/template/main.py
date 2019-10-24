
import dataset
from model import LeNet5, CustomMLP

# import some packages you need here
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.train()
    trn_loss = 0.0
    corr_num = 0
    total_num = 0
    for i, data in enumerate(trn_loader):
        x, label = data
        if device:
            x = x.cuda()
            label = label.cuda()
        # grad init
        optimizer.zero_grad()
        # forward propagation
        model_output = model(x)
        # calculate loss
        loss = criterion(model_output, label)
        # back propagation 
        loss.backward()
        # weight update
        optimizer.step()
        
        model_label = model_output.argmax(dim=1)
        corr = model_label[model_label == model_label].size(0)
        corr_num += corr
        total_num += model_label.size(0)
        
        # trn_loss summary
        trn_loss += loss.item()
        # del (memory issue)
        del loss
        del model_output
        
        # 학습과정 출력
    acc = (corr_num / total_num * 100)
    print(" trn loss: {:.4f} | trn accuracy: {:.4f}".format(
             trn_loss / len(trn_loader), acc
        ))
    
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    with torch.no_grad(): 
        tst_loss = 0.0
        corr_num = 0
        total_num = 0
        for j, val in enumerate(tst_loader):
            val_x, val_label = val
            if device:
                val_x = val_x.cuda()
                val_label =val_label.cuda()
            val_output = model(val_x)
            v_loss = criterion(val_output, val_label)
            tst_loss += v_loss
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)
            
            del val_output
            del v_loss
    acc = (corr_num / total_num * 100)
    print(" tst loss: {:.4f} | tst accuracy: {:.4f}".format(
             tst_loss / len(tst_loader), acc
        ))
    
    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    batch_size=64
    trn_dataset = dataset.MNIST(data_dir='../data/train')
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
    tst_dataset = dataset.MNIST(data_dir='../data/test')
    tst_loader = torch.utils.data.DataLoader(tst_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
    
    LeNet = LeNet5().to(device)
    Custom = CustomMLP().to(device)

    # loss
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(LeNet.parameters(), lr=0.01, momentum=0.9)
    optimizer2 = optim.SGD(Custom.parameters(), lr=0.01, momentum=0.9)
    # hyper-parameters
    num_epochs = 20       
    trn_loss_list1 = []
    tst_loss_list1 = []
    trn_acc1 = []
    tst_acc1 = []
    trn_loss_list2 = []
    tst_loss_list2 = []
    trn_acc2 = []
    tst_acc2 = []
    for epoch in range(num_epochs):
        print("epoch : ", (epoch+1))
        print("LeNet")
        trn_loss, trn_acc = train(LeNet, trn_loader, device, criterion, optimizer1)
        trn_loss_list1.append(trn_loss / len(trn_loader))
        trn_acc1.append(trn_acc)
        tst_loss, tst_acc = test(LeNet, tst_loader, device, criterion)
        tst_loss_list1.append(tst_loss / len(tst_loader))
        tst_acc1.append(tst_acc)
        print("CustomMLP")
        trn_loss, trn_acc = train(Custom, trn_loader, device, criterion, optimizer2)
        trn_loss_list2.append(trn_loss / len(trn_loader))
        trn_acc2.append(trn_acc)
        tst_loss, tst_acc = test(Custom, tst_loader, device, criterion)
        tst_loss_list2.append(tst_loss / len(tst_loader))
        tst_acc2.append(tst_acc)
    
        
    plt.figure(figsize=(5,4))
    x_range = range(len(trn_loss_list1))
    plt.plot(x_range, trn_loss_list1, label="trn")
    plt.plot(x_range, tst_loss_list1, label="tst")
    plt.legend()
    plt.ylim(0, 1)
    plt.title('LeNet-5 Loss')
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.grid()
    
    plt.figure(figsize=(5,4))
    x_range = range(len(trn_acc1))
    plt.plot(x_range, trn_acc1, label="trn")
    plt.plot(x_range, tst_acc1, label="tst")
    plt.legend()
    plt.ylim(0, 100)
    plt.title('LeNet-5 Accuracy')
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.grid()

    plt.figure(figsize=(5,4))
    x_range = range(len(trn_loss_list2))
    plt.plot(x_range, trn_loss_list2, label="trn")
    plt.plot(x_range, tst_loss_list2, label="tst")
    plt.legend()
    plt.ylim(0, 1)
    plt.title('CustomMLP Loss')
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.grid()

    
    plt.figure(figsize=(5,4))
    x_range = range(len(trn_acc2))
    plt.plot(x_range, trn_acc2, label="trn")
    plt.plot(x_range, tst_acc2, label="tst")
    plt.legend()
    plt.ylim(0, 100)
    plt.title('CustomMLP Accuracy')
    plt.xlabel("training steps")
    plt.ylabel("acc")
    plt.grid()
        
    # val acc
    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(tst_loader):
            val_x, val_label = val
            if device:
                val_x = val_x.cuda()
                val_label =val_label.cuda()
            val_output = LeNet(val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)
    
    print("LeNet5 acc: {:.2f}".format(corr_num / total_num * 100))
    
    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(tst_loader):
            val_x, val_label = val
            if device:
                val_x = val_x.cuda()
                val_label =val_label.cuda()
            val_output = Custom(val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)
    
    print("CustomMLP acc: {:.2f}".format(corr_num / total_num * 100))
        
    
if __name__ == '__main__':
    main()
