import os
import pandas as pd
import torch
import numpy as np
import sklearn.preprocessing as preprocessor
import xlsxwriter
import pathlib
import json
import config
import pickle
import torch.nn as nn

def load_data(
              dataset_path: str,
              task: str,
              dataset_num: str
              ):
        
        drop_columns = ['s1','s5','s6','s10','s16','s18','s19']
        file = pd.read_csv(os.path.join(dataset_path,
                                        f'{task}_{dataset_num}.csv')).drop(['setting1',
                                                                             'setting2',
                                                                             'setting3'],axis=1)
        
        file_list = []
        cycle_list = []
        grouped_file = file.groupby(file.engine_id)
        total_eng = file['engine_id'].iloc[-1]
        file = file.drop(drop_columns,axis=1, inplace=True)
        for i in range(total_eng):
            
            i = i+1
            file_grouped = grouped_file.get_group(i)
            cycle_list.append(file_grouped['cycle'].iloc[-1])
            file_list.append(file_grouped.drop(['engine_id','cycle'],axis=1))
    
        return file_list , cycle_list, total_eng
    
def labels_generator(
                     cycle_list: list,
                     train: bool,
                     data_length,
                     dataset_path,
                     total_eng,
                     dataset_num
                     ):

        labels = []
        ext_rul = pd.read_csv(os.path.join(dataset_path, f'RUL_{dataset_num}.csv')).values
        start_rul = 120
        for i in range(total_eng):
            
            if train:
                
                total_rul = cycle_list[i]
            else:
                
                total_rul = cycle_list[i] + ext_rul[i].item()
                
            healthy_cyc = total_rul - start_rul
            
            degrading_cyc = total_rul - healthy_cyc
            
            healthy_rul = start_rul * np.ones((healthy_cyc,1))
            
            degrading_rul = np.linspace(start = start_rul -1 ,
                                        stop = 0,
                                        num = degrading_cyc)
            
            degrading_rul = np.expand_dims(degrading_rul, axis = 1)
            
            final_rul = np.concatenate((healthy_rul,degrading_rul), axis = 0)
            
            if train:
                
                final_rul = final_rul
            else:
                
                final_rul = final_rul[:cycle_list[i],:]
                
            if data_length[i]<final_rul.shape[0]:
                
                if (train == False and ext_rul[i].item()>=start_rul) : 
                    
                    final_rul = final_rul[:data_length[i]] 
                else:
                    
                    final_rul = final_rul[-data_length[i]:]
                
            labels.append(torch.Tensor(final_rul))
            
        return torch.cat(labels,dim=0)
    
def scale_data(
               data,
               data_scale_params,
               scaling_method
               ):
        
        if scaling_method == '-11':
            
            scaler = preprocessor.MinMaxScaler(feature_range = (-1,1))
        elif scaling_method == '01':
            
            scaler = preprocessor.MinMaxScaler(feature_range = (0,1))
        else:
            
            scaler = preprocessor.StandardScaler()
        
        if data_scale_params is None:
            
            data_scale_params = scaler.fit(pd.concat(data , axis = 0))

        scaled_data = [torch.FloatTensor(data_scale_params.transform(x)) for x in data]
        
        return scaled_data, data_scale_params

def window_data(
                dataset,
                window_length,
                window_step
                ):
        
        windowed_data_list = []
        
        for data in dataset:
            
            windowed_data_list.append(data.unfold(0,window_length,window_step))
            
        eng_id_lengths = [i.shape[0] for i in windowed_data_list]
        
        return torch.cat(windowed_data_list, dim = 0), eng_id_lengths
    
def model_loader(path):
    
    with open(path, 'r') as file:
            hyperparams = json.load(file)
    best_trial_num = hyperparams['Trial_Num']
    ckpt = torch.load(os.path.join(pathlib.Path(path).parents[1], 
                                   'Num_Trial '+str(best_trial_num),
                                   'Model.pt'),map_location=config.DEVICE)
    model = ckpt['network']
    d_model = hyperparams['d_model']
    '''
    if encoder:
        del model.rul_predictor
        encoder = nn.Sequential(model.linear_embedding,
                                model.transformer)
        return encoder, d_model
    else:
        
        decoder = model.rul_predictor

        return decoder
   
    '''
    return model, d_model
        
def score_cal(
              test_preds,
              test_labels,
              test_lengths,
              inference
              ):
    
        test_preds = torch.split(test_preds,test_lengths)
        test_labels = torch.split(test_labels,test_lengths) 
        score_list = []
        pred_list = []
        label_list = []
        a1 = 13
        a2 = 10
        
        for i in range(len(test_preds)):
            
            pred_RUL = test_preds[i][-1] 
            actual_RUL = test_labels[i][-1]
            pred_list.append(pred_RUL)
            label_list.append(actual_RUL)
            d = pred_RUL - actual_RUL
            
            if d < 0:
                
                score_list.append((torch.exp(-(d/a1)) - 1).item())
                
            else:
                
                score_list.append((torch.exp((d/a2)) - 1).item())
                
        if inference:
            return sum(score_list)
        else:
            return sum(score_list),score_list, pred_list, label_list
        
def param_results(best_trial,result_excel):
        
        best_para = os.path.join(result_excel, 'Best Trial info.txt')
        best_trial.params['Loss'] = round(best_trial.user_attrs['loss'],2)
        best_trial.params['Score'] = int(best_trial.user_attrs['score'])
        best_trial.params['Epoch'] = best_trial.user_attrs['epoch']
        best_trial.params['Trial_Num'] = best_trial.number
        with open(best_para, 'w') as file:
            json.dump(best_trial.params, file)
        
def results_dataframe(results_df, 
                      result_excel, 
                      best_epoch, 
                      best_loss):
        
        converter = pd.ExcelWriter(os.path.join(result_excel, 
                                                'Result Dataframe.xlsx'))
        converter_2 = pd.ExcelWriter(os.path.join(result_excel, 
                                                 'Result Dataframe Sorted.xlsx'))
        results_df.to_excel(converter)
        converter.save()
        results_df = results_df.drop(results_df[results_df.state == 'PRUNED'].index)
        results_df['Test Loss'] = best_loss
        results_df['Epoch'] = best_epoch
        results_df = results_df.drop(['datetime_start',
                                      'datetime_complete',
                                      'duration'], axis=1)
        results_df = results_df.sort_values(by = 'value', ascending=True)
        results_df.to_excel(converter_2)
        converter_2.save()
        print('Results Dataframe exported')

def model_saver(path, 
                trg_model, 
                discr_model,
                trial,
                score,
                loss,
                epoch):
            
            trial.set_user_attr('epoch', epoch)
            trial.set_user_attr('loss', loss)
            trial.set_user_attr('score', score)
            trg_save = {'network': trg_model,
                        'state_dict': trg_model.state_dict()}
            
            discr_save = {'network': discr_model,
                          'state_dict': discr_model.state_dict()}
            
            torch.save(trg_save, path + '/trg_model.pt')
            torch.save(discr_save, path + '/dis_model.pt')
            
def learning_rate_excel(lr_list,
                        epoch_list,
                        loss_list, 
                        result_path):
        
        row_num = 1
        lr_workbook = xlsxwriter.Workbook(os.path.join(result_path, 
                                                       'Learning Rates.xlsx'))
        worksheet = lr_workbook.add_worksheet()
        bold = lr_workbook.add_format({'bold': True})
        worksheet.write('A1', 'Epochs',bold)
        worksheet.write('B1', 'Training Loss',bold)
        worksheet.write('C1', 'Batchwise Learning Rate',bold)
        for value in range(len(epoch_list)):
            worksheet.write(row_num,0, epoch_list[value])
            worksheet.write(row_num,1, loss_list[value])
            for lr in range(len(lr_list[value])):
                worksheet.write(row_num,2, lr_list[value][lr])
                row_num = row_num+1
        lr_workbook.close()
        
def accuracy_excel(test_score_list,
                   epoch, 
                   test_loss, 
                   prediction_list, 
                   target_list, 
                   sum_score,
                   result_path):
            
            accuracy_workbook = xlsxwriter.Workbook(os.path.join(result_path, 
                                                                 'Accuracy Scores.xlsx'))
            worksheet = accuracy_workbook.add_worksheet()
            formating = accuracy_workbook.add_format({'bold': True,
                                                   'align': 'center',
                                                   'border': 2})
            worksheet.write('A1', 'Epochs',formating)
            worksheet.write('B1', 'Test Loss', formating)
            worksheet.write('C1', 'Sum Score' , formating)
            worksheet.write('D1', 'Engine Id' , formating)
            worksheet.write('E1', 'Prediction' , formating)
            worksheet.write('F1', 'Label' , formating)
            worksheet.write('G1', 'Engine Scores' , formating)
            worksheet.write('A2', epoch)
            worksheet.write('B2', test_loss)
            worksheet.write('C2', sum_score)
            for i in range(len(prediction_list)):
                
                worksheet.write(i+1,3,i+1)
                worksheet.write(i+1,4,prediction_list[i])
                worksheet.write(i+1,5,target_list[i])
                worksheet.write(i+1,6,test_score_list[i])
            accuracy_workbook.close()
        
class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def get_learning_rate(self):
        factor = 0.5 * self.d_model ** (-0.5)
        
        
        if self.step_num < self.warmup_steps:
            scale = 0.25 * (self.step_num*self.warmup_steps**(-1.5))
        else:
            scale = 0.25 * (self.warmup_steps*self.warmup_steps**(-1.5))

        return factor * scale
    
def inference_loader(path,
                 n_trial
                 ):
    
    trg_model_path = os.path.join(path,'Num_Trial ' + str(n_trial), 'trg_model.pt')
    discr_model_path = os.path.join(path,'Num_Trial ' + str(n_trial), 'dis_model.pt')
    latent_space_data_path = os.path.join(path,'Num_Trial ' + str(n_trial), 'Latent Space Data.pkl')
    
    ckpt = torch.load(trg_model_path)
    ckpt_discr = torch.load(discr_model_path)
    trg_model = ckpt['network']
    discr_model = ckpt_discr['network']
    
    if all(config.latent_space_conditions):
        with open(latent_space_data_path, 'rb') as file:
            latent_space_data = pickle.load(file)
    else:
        latent_space_data=None

    return trg_model, discr_model, latent_space_data

def path_creator(self,trial_num,main_path,time_stamp):
    
    if config.use_gm:
        project_name = 'ADDA_GM'
        gm_method = config.gm_method + '_' + config.deci_mat_method
    else:
        project_name = 'ADDA'
        gm_method = ''
        
    result_path = os.path.join(main_path.parents[1],'sciebo',
                               'Hussam_Aleem_MasterThesis',
                               '00_Results',
                               project_name,
                               gm_method,
                               time_stamp + '_' + 
                               config.src_dataset + '_' +
                               config.trg_dataset,
                               config.net,
                               'Num_Trial ' + str(trial_num))

    paths = []    
    main_result_dir = pathlib.Path(result_path).parent
    result_excel = os.path.join(main_result_dir, ('Detail Result Files'))
    

    loader_path = os.path.join(main_path.parents[1],
                               'sciebo',
                               'Hussam_Aleem_MasterThesis',
                               '00_Results',
                               'Baseline')
    
    folders = os.listdir(loader_path)
    
    for folder in folders:
        if config.src_dataset in folder:
            trial_num_loader_path = os.path.join(loader_path, 
                                                 folder,
                                                 config.net+'_'+'Model',
                                                 'Detail Result Files',
                                                 'Best Trial info.txt')

    
    rul_plot_path = os.path.join(main_result_dir, 'Inference Plots')
    gif_images_path = os.path.join(main_result_dir, 'Latent Space Plots GiF')
    latent_space_plot_path = os.path.join(main_result_dir, 'Latent Space Plots')
    if all(config.latent_space_conditions):
        paths.extend([result_path,result_excel,rul_plot_path, latent_space_plot_path, gif_images_path])
    else:
        paths.extend([result_path,result_excel,rul_plot_path])
    for i in paths:
        
        if not os.path.exists(i):
            
            os.makedirs(i)
            
    return result_path,result_excel,rul_plot_path,trial_num_loader_path, latent_space_plot_path, gif_images_path

def latent_space_data_saver(latent_space_data,best_epoch,path):
    config.latent_space_epochs.append(best_epoch+1)
    epochs_to_delete = [k for k in latent_space_data.keys() if k not in config.latent_space_epochs]
    for key in epochs_to_delete:
        del latent_space_data[key]
    with open(os.path.join(path, 'Latent Space Data.pkl'), 'wb') as file:

        pickle.dump(latent_space_data,file)
    config.latent_space_epochs.remove(best_epoch+1)
    
def calculate_mmd_loss(src_feat, trg_feat, kernel_type='gaussian', kernel_bandwidth=1.0):
    src_mean = torch.mean(src_feat, dim=0)
    trg_mean = torch.mean(trg_feat, dim=0)

    src_kernel = compute_kernel(src_feat, src_mean, kernel_type, kernel_bandwidth)
    trg_kernel = compute_kernel(trg_feat, trg_mean, kernel_type, kernel_bandwidth)
    cross_kernel = compute_kernel(src_feat, trg_feat, kernel_type, kernel_bandwidth)

    mmd_loss = torch.mean(src_kernel) + torch.mean(trg_kernel) - 2 * torch.mean(cross_kernel)
    return mmd_loss

def compute_kernel(x, y, kernel_type='gaussian', kernel_bandwidth=1.0):
    if kernel_type == 'gaussian':
        euli_dist = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2)
        kernel_val = torch.exp(-euli_dist / (2 * kernel_bandwidth ** 2))
    return kernel_val

