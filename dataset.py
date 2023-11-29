from torch.utils.data import Dataset
import os
import utils

class CmapssDataset(Dataset):
    
    def __init__(self, 
                 root: str , 
                 train: bool = True,
                 dataset_num: str = 'FD001', 
                 window_step: int =  20, 
                 data_scale_params = None,
                 scaling_method: str = '-11',
                 bsc: bool = False,
                 window_length: int = 50,
                 ):
        
        self.data_scale_params = data_scale_params
        
        if train:
            task = 'train'
        else:
            task = 'test'
        
        dataset_path = os.path.join(root, 'Datasets',
                                            'CMAPSS')


        dataset , cycle_list, total_eng = utils.load_data(dataset_path = dataset_path, 
                                                          task = task,
                                                          dataset_num = dataset_num 
                                                          )
        
        scaled_data, self.data_scale_params = utils.scale_data(data = dataset,
                                                               data_scale_params = self.data_scale_params, 
                                                               scaling_method = scaling_method)
        
        windowed_data , self.eng_id_lengths = utils.window_data(dataset = scaled_data, 
                                                                window_length = window_length,
                                                                window_step = window_step)
        
        
        labels = utils.labels_generator(cycle_list = cycle_list,  
                                        train = train,
                                        data_length = self.eng_id_lengths,
                                        dataset_path = dataset_path,
                                        total_eng = total_eng,
                                        dataset_num = dataset_num
                                        )
            
        
        
        if bsc:
            
            windowed_data = windowed_data.transpose(1,2)
            

            
        self.x , self.y = windowed_data , labels
        print(dataset_num, self.x.shape)
        
    def __getitem__(self, index):
        return self.x[index] , self.y[index]
        
    def __len__(self):
        return len(self.x)