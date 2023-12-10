import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms

class  GalaxyDataset(Dataset):
    def __init__(self,csv_file,image_dir,transform=None):
        super().__init__()
        """Args:
        
           csv_file: path to the label csv
           image_dir : path  to the dir containing all images
           transform: transform to apply
        
        """
        self.label_df = pd.read_csv(csv_file)
        self.label_df = self.label_df[["galaxyID","label1"]].copy()
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        """
        Return the size of the dataset
        
        """    
        num_files = len([i for i in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, i))])
        return len(self.label_df),num_files
    def __getitem__(self, index):
        """
        gth the index-th sample.
          output the image(chw)and the ture label

        """
        if torch.is_tensor(index):
            index = index.tolist()


        galaxyID = self.label_df.iloc[index,0].astype(str)
        
        image_path = os.path.join(self.image_dir,galaxyID +".jpg")
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.label_df.iloc[index,1]) 
        return image,label,int(galaxyID)   
    
    def Transform(is_for_inception=False):
        """
        
        create pytorch data trabsfirns for galaxy dagaset.
        Args:
        is_for_inception(bool):True for inception nural network
        outputs:
        train_transform:transform for the training data
        test_transform: transform for the testing data
        
     """
        if is_for_inception :
            input_size = 299
        else:
            input_size =224

        train_transform = transforms.Compose([
                transforms.CenterCrop(input_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(input_size,scale=(0.8,1.0),ratio =(0.99,1.01)),

                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[[0.5,0.5,0.5]])


                    ])        
        valid_transform = transforms.Compose([
                                           transforms.CenterCrop(input_size),

                                            transforms.Normalize([0.5,0.5,0.5],[[0.5,0.5,0.5]])



               ])
        test_transform = transforms.Compose([
                                           transforms.CenterCrop(input_size),

                                            transforms.Normalize([0.5,0.5,0.5],[[0.5,0.5,0.5]])



               ])