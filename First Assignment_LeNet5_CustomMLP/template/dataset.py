
# import some packages you need here
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import os
import tarfile
tf = tarfile.open("../data/train.tar")
tf.extractall('../data/')
tf.close()
tf = tarfile.open("../data/test.tar")
tf.extractall('../data/')
tf.close()


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):

        # write your codes here
        self.data_dir = data_dir
        images = []
        all_categories = []
        for filename in os.listdir(data_dir):
            img = cv2.imread(os.path.join(data_dir,filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
            category = int(filename.split('_')[-1].split('.')[0])
            all_categories.append(category)
        self.imgfile = images
        self.Label = all_categories
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomRotaion(10.),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = data_transform
    def __len__(self):

        # write your codes here
        return len(self.imgfile)

    def __getitem__(self, idx):

        # write your codes here
        img = self.imgfile[idx]
        label = self.Label[idx]
        
        img = self.transform(img)
        
        
        return img, label

if __name__ == '__main__':

    # write test codes to verify your implementations
    test_dataset = MNIST(data_dir='../data/test')
    img, label = test_dataset[0]
    img.size()                                

    

