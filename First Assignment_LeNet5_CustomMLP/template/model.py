
import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):

        # write your codes here
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1) # 6@24*24
        # activation ReLU
        self.pool1 = nn.MaxPool2d(2) # 6@12*12
        self.conv2 = nn.Conv2d(6, 16, 5, 1) # 16@8*8
        # activation ReLU
        self.pool2 = nn.MaxPool2d(2) # 16@4*4
                
        self.fc1 = nn.Linear(16*4*4, 120)
        # activation ReLU
        self.fc2 = nn.Linear(120, 84)
        # activation ReLU
        self.fc3 = nn.Linear(84, 10)

        
    def forward(self, img):

        # write your codes here
        # convolve, then perform ReLU non-linearity
        img = nn.functional.relu(self.conv1(img))  
        # max-pooling with 2x2 grid 
        img = self.pool1(img) 
        # convolve, then perform ReLU non-linearity
        img = nn.functional.relu(self.conv2(img))
        # max-pooling with 2x2 grid
        img = self.pool2(img)
        
        # make linear
        dim = 1
        for d in img.size()[1:]: #16, 4, 4
            dim = dim * d
        img = img.view(-1, dim)
        # FC-1, then perform ReLU non-linearity
        img = nn.functional.relu(self.fc1(img))
        # FC-2, then perform ReLU non-linearity
        img = nn.functional.relu(self.fc2(img))
        # FC-3
        img = self.fc3(img)
        output = img
        #output = F.softmax(out, dim=1)

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):

        # write your codes here
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 71)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(71, 71)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(71, 10)

    def forward(self, img):

        # write your codes here
        img = img.view(-1, 28*28)
        img = nn.functional.relu(self.fc1(img))
        img = self.fc1_drop(img)
        img = nn.functional.relu(self.fc2(img))
        img = self.fc2_drop(img)
        img = self.fc3(img)
        
        output = img

        return output
