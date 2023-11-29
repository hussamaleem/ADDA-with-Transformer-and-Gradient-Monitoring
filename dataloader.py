import dataset
from torch.utils.data import DataLoader, Sampler
import random
import config


def load_datasets(window_length, dataset_num, main_path):
    
    train_dataset  = dataset.CmapssDataset(
                                root = main_path.parents[0], 
                                train = True,
                                dataset_num = dataset_num,
                                window_step = config.window_step,
                                data_scale_params = None,
                                scaling_method = config.scaling_method,
                                bsc = config.bsc,
                                window_length = window_length,
                                        )
    
    test_dataset = dataset.CmapssDataset(
                                root = main_path.parents[0], 
                                train = False,
                                dataset_num = dataset_num,
                                window_step = config.window_step,
                                data_scale_params = train_dataset.data_scale_params,
                                scaling_method = config.scaling_method,
                                bsc = config.bsc,
                                window_length = window_length,
                                        )
    test_lengths = test_dataset.eng_id_lengths
    
    return (train_dataset, test_dataset), test_lengths

class RepeatSampler(Sampler):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = len(dataset2)
    
    def __iter__(self):
        index_list = []
        for i in range(self.length):
            index_list.append(i % len(self.dataset1))
        random.shuffle(index_list)
        return iter(index_list)
    
    def __len__(self):
        return self.length
    
def dataloaders(train_dataset = None, 
                test_dataset = None,  
                batch_size = None, 
                sampler = None, 
                shuffle: bool = False):
    
    train_loader = DataLoader(train_dataset, 
                              batch_size = batch_size, 
                              sampler = sampler)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, 
                                 batch_size = batch_size, 
                                 shuffle = False)
    else: 
        test_loader = None

    return train_loader, test_loader
