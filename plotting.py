import matplotlib.pyplot as plt
import torch
import config
from sklearn.manifold import TSNE
import os
import glob
from PIL import Image


class train_test_plots():
    
    def __init__(self,
                 dataset_num: str,
                 result_path,
                 test_lengths,
                 rul_plot_path,
                 latent_space_plot_path
                 ):
        self.dataset_num = dataset_num
        self.result_path = result_path
        self.test_lengths = test_lengths
        self.rul_plot_path = rul_plot_path
        self.latent_space_plot_path = latent_space_plot_path
        self.tsne = TSNE(n_components=2)
            
            
    def loss_plots(self,loss,train,name):
        
        if train:
            name = name
        else:
            name = 'Validation Loss'
            
        plt.plot(loss, 'r', linewidth = 2.5,
                 label = "Loss")
        plt.grid()  
        plt.ylabel(f'{name}', size = 15)
        plt.xlabel('Epochs', size = 15)
        plt.savefig(self.result_path + f'/{name}', 
                    bbox_inches = 'tight', dpi=150)
        plt.close()
        
    def testing_plots(self,preds,labels):
        
        preds = torch.split(preds,self.test_lengths)
        labels = torch.split(labels,self.test_lengths)
        for i in range(len(preds)):
            
            plt.plot(preds[i], 'r', linewidth = 2.5, 
                     label = "Predicted RUL")
            plt.plot(labels[i], 'b', linewidth = 2.5, 
                     label = "Actual RUL")
            plt.legend(loc='upper right')
            plt.grid()  
            plt.ylabel('Remaining Useful Life', size = 15)
            plt.xlabel('Cycles', size = 15)
            plt.title(f'{self.dataset_num} Test Engine ID {i+1}', size = 20)
            plt.savefig(self.rul_plot_path + 
                        f'/Testing RUL plot, Eng_id {i+1}', 
                        bbox_inches = 'tight', dpi=150)
            plt.close('all')
            plt.close()
            
    def latent_space_plot(self,latent_space_data):
        for i in latent_space_data.keys():
           feat = latent_space_data[i]['Features']
           labels = latent_space_data[i]['Label']
           for k in range(len(feat)):
               transformed_feat = self.tsne.fit_transform(feat[k])
               fig, ax = plt.subplots(figsize=(10, 10))
               scatter = ax.scatter(x=transformed_feat[:, 0], y=transformed_feat[:, 1], s=2.0, 
                                    c=labels[k], cmap='tab10', alpha=0.9, zorder=2)
       
               ax.spines["right"].set_visible(False)
               ax.spines["top"].set_visible(False)
               ax.set_title(f'Latent Space Plot - Epoch {i}, Iteration {k}')
               plt.savefig(self.latent_space_plot_path + f'/Epoch {i} , iteration {k}',bbox_inches = 'tight', dpi=150)
               plt.close()
    
    def create_latent_space_gif(self,plot_path,gif_path,best_epoch):
        width = 1251
        height = 1197
        config.latent_space_epochs.append(best_epoch)
        for epoch in config.latent_space_epochs:
            image_files = []
            file_pattern = f'/Epoch {epoch} , iteration *'
            files = glob.glob(plot_path + file_pattern)
            files.sort(key=lambda x: int(x.split('iteration ')[1].split('.')[0]))
            for i in files:
                image = Image.open(i)
                resized_image = image.resize((width, height))
                image_files.append(resized_image)
            gif_file = os.path.join(gif_path,f'Epoch {epoch}.gif')
            image_files[0].save(gif_file, format='GIF', append_images=image_files[1:], save_all=True, duration=500, loop=1)
            
            