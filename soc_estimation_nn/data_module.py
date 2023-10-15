
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl


class DataModule(pl.LightningDataModule):


    def __init__(
            self, 
            dataset: Dataset, 
            train_split: float = 0.7, 
            val_split: float = 0.2,
            test_split: float = 0.1
        ):
        super().__init__()
        self.dataset = dataset
        
        if round(sum([train_split, val_split, test_split]), 6) != 1:
            raise ValueError('All of train/val/test splits must sum up to 1.')
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split


    def setup(self, stage: str):
        num_dataset = len(self.dataset)
        num_training_set = round(self.train_split * num_dataset)
        num_validation_set = round(self.val_split * num_dataset)
        num_test_set = round(self.test_split * num_dataset)

        self.training_set, self.validation_set, self.test_set = random_split(
            self.dataset, [num_training_set, num_validation_set, num_test_set]
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_set, batch_size=1, shuffle=True)
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_set, batch_size=1, shuffle=False)
    

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=1, shuffle=False)