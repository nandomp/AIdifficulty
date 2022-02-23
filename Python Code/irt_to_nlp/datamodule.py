

from typing import Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from irt_to_nlp.config import Dataset
from irt_to_nlp.loader_nlp import Loader


from irt_to_nlp.config import ModelAvailable
class NLPDataModule(LightningDataModule):
    """
     A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    """
    
    def __init__(self, 
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 pin_memory:bool,
                 dataset:Dataset,
                 model_name:str,
                 train_val_test_split_percentage:Tuple[float,float,float]=(0.7,0.3,0.0),
                 
                 
                 
                 ):
        super().__init__()
        self.data_dir=data_dir
        self.data_dir = data_dir
        self.train_val_test_split_percentage = train_val_test_split_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_enum=dataset
        self.dataset=Loader
        self.model=ModelAvailable[model_name]
        
    def prepare_data(self):
        """Se necesita el csv que proporciona Nando"""
        
        pass
    
    def setup(self,stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        fulldataset = self.dataset(self.data_dir,self.model)
        train_val_test_split= [round(split*len(fulldataset)) for split in self.train_val_test_split_percentage]
        
        
        if not sum(train_val_test_split)==len(fulldataset):
            train_val_test_split[0]+=1
        self.data_train, self.data_val, self.data_test = random_split(
            fulldataset, train_val_test_split
        )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
