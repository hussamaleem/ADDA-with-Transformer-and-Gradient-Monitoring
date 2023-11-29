import optuna
import hyperparams
import adda_models
import dataloader as loader
import gradient_monitoring
import trainer
import torch.nn as nn
import utils
from optuna.trial import TrialState
import pathlib
import os
import plotting
import torch.optim as optim
import config
from tqdm import trange
import math
import json
from itertools import cycle


class OptunaOptim():
    
    def __init__(self,
                 main_path,
                 src_dataset_num,
                 trg_dataset_num,
                 net,
                 window_type,
                 deci_mat_method,
                 gm_method,
                 initiator,
                 device,
                 use_gm,
                 time_stamp
                 ):
        
        self.main_path = main_path
        self.net = net
        self.src_dataset_num = src_dataset_num
        self.trg_dataset_num = trg_dataset_num
        self.time_stamp = time_stamp
        self.window_type = window_type
        self.deci_mat_method = deci_mat_method
        self.gm_method = gm_method
        self.initiator = initiator
        self.best_loss_list = []
        self.best_epoch_list = []
        self.device = device
        self.use_gm = use_gm
        
    
    def get_hyperparams(self,trial):
        self.params = hyperparams.Hyperparameters(trial=trial,
                                                  gm_method=self.gm_method,
                                                  use_gm=self.use_gm,
                                                  use_custom_scheduler=config.use_custom_scheduler).get_hyperparams()
        
        
    def create_parameters(self,trial):
        
        self.discr_lr = self.params['Discr_Learning_Rate']
        
        self.trg_encoder_LR = self.params['Trg_Encoder_LR']
        
        self.batch_size = config.batch_size
        
        self.hidden_size = self.params['Hidden_Size']
        
        self.n_epochs = self.params['Num_Epochs']
        
        self.warm_up_ratio = self.params['Warm_Up_Ratio']
        
        if self.use_gm:
            self.learn_fact = self.params['Learning_Factor']
            if 'Vanilla' in self.gm_method:
                self.start_epoch = self.params['Start_Epoch']
                self.repeat_epoch = self.params['Repeating_Epoch']
                
            elif 'Momentum' in self.gm_method:
                self.momentum = self.params['Momentum'] 
                self.alpha = None

            else:
                self.momentum = self.params['Momentum'] 
                self.alpha = self.params['Alpha']
            
    def load_models(self):

        base_model, self.bottleneck_dim = utils.model_loader(path=self.trial_num_loader_path)
        self.src_encoder = adda_models.ADDAEncoder(base_model=base_model,src_encoder=True).get_encoder()
        self.trg_encoder = adda_models.ADDAEncoder(base_model=base_model,src_encoder=False).get_encoder()
        self.rul_predictor = adda_models.ADDARULPredictor(base_model=base_model).get_decoder()
        self.discriminator = adda_models.Discriminator(in_features=self.bottleneck_dim, 
                                                       out_features=self.hidden_size
                                                       ).to(self.device)
    def get_loaders(self):
        
        self.src_window_length = config.window_length[(int(self.src_dataset_num[4])-1)]
        self.trg_len = config.window_length[(int(self.trg_dataset_num[4])-1)]
        
        src_datasets, _ = loader.load_datasets(window_length=self.src_window_length, 
                                               dataset_num=self.src_dataset_num, 
                                               main_path=self.main_path)
        src_train_dataset, _ = src_datasets
        
        trg_datasets, self.test_lengths = loader.load_datasets(window_length=self.trg_len,
                                                               dataset_num=self.trg_dataset_num, 
                                                               main_path=self.main_path)
        trg_train_dataset, trg_test_dataset = trg_datasets
        
        self.src_train_loader,_ = loader.dataloaders(src_train_dataset, batch_size=config.batch_size)
        self.trg_train_loader,self.trg_test_loader = loader.dataloaders(train_dataset=trg_train_dataset,
                                                                        test_dataset=trg_test_dataset,
                                                                        batch_size=config.batch_size)
        max_len = max([src_train_dataset.x.shape[0], trg_train_dataset.x.shape[0]])
        if src_train_dataset.x.shape[0] == max_len:
            self.trg_train_loader = cycle(self.trg_train_loader)
            self.total_iter = len(src_train_dataset)
        else:
            self.src_train_loader = cycle(self.src_train_loader)
            self.total_iter = len(trg_train_dataset)
 
    def training_setup(self):
        warm_up_steps = math.ceil(self.total_iter/config.batch_size)*config.n_epochs
        
        self.optimizer_discr = optim.Adam(self.discriminator.parameters(), 
                                             lr=self.discr_lr, 
                                             weight_decay=1e-3)
        self.optimizer_trg = optim.Adam(self.trg_encoder.parameters(), 
                                           lr=self.trg_encoder_LR,
                                           betas=(0.9,0.98),
                                           eps=1e-09) 
        if config.use_custom_scheduler:
            self.lr_sched_trg =  utils.TransformerScheduler(self.optimizer_trg, 
                                                            d_model = self.bottleneck_dim, 
                                                            warmup_steps = int(warm_up_steps*self.warm_up_ratio))
        else:
            self.lr_sched_trg = optim.lr_scheduler.OneCycleLR(self.optimizer_trg, 
                                                              max_lr = 0.01,
                                                              epochs=self.n_epochs,
                                                              steps_per_epoch=math.ceil(self.total_iter/config.batch_size))
        

        self.lr_sched_discr = optim.lr_scheduler.OneCycleLR(self.optimizer_discr, 
                                                            max_lr = 0.01,
                                                            epochs=self.n_epochs,
                                                            steps_per_epoch=math.ceil(self.total_iter/config.batch_size))

        self.criterion_discr = nn.CrossEntropyLoss()
        self.criterion_trg = nn.MSELoss()
        
        if self.use_gm:
            self.gm = gradient_monitoring.GradientMonitoring(trg_encoder=self.trg_encoder, 
                                                             learning_factor=self.learn_fact, 
                                                             momentum=self.momentum, 
                                                             mome_mat_initiator=self.initiator, 
                                                             decision_matrix=self.deci_mat_method,
                                                             device=self.device,
                                                             alpha=self.alpha)
    def objective(self,trial):  
        
        num = trial.number
        test_score_list = []
        discr_training_loss = []
        total_trg_encoder_loss = []
        training_epoch_list = []
        lr_list = []
        self.test_loss_list = []
        self.best_epoch = 0
        self.best_loss = 0
        self.latent_space_data = {}

        self.result_path,self.result_excel,self.rul_plot_path,self.trial_num_loader_path,self.latent_space_plot_path, self.gif_images_path = utils.path_creator(self, 
                                                                                                                                                                trial_num = num, 
                                                                                                                                                                main_path=self.main_path, 
                                                                                                                                                                time_stamp=self.time_stamp)

        self.get_hyperparams(trial=trial)
        self.create_parameters(trial)
        self.load_models()
        self.get_loaders()
        self.training_setup()
        
        self.train_test_class = trainer.Trainer(src_encoder=self.src_encoder, 
                                                trg_encoder=self.trg_encoder, 
                                                discr=self.discriminator, 
                                                device=self.device, 
                                                discr_optimizer=self.optimizer_discr, 
                                                trg_optimizer=self.optimizer_trg, 
                                                discr_criterion=self.criterion_discr, 
                                                trg_criterion=self.criterion_trg, 
                                                rul_predictor=self.rul_predictor,
                                                gm = self.gm if self.use_gm else None,
                                                gm_method=self.gm_method,
                                                lr_sched_trg=self.lr_sched_trg,
                                                lr_sched_discr=self.lr_sched_discr,
                                                use_gm=self.use_gm) 
        
        self.plot = plotting.train_test_plots(dataset_num=self.trg_dataset_num, 
                                              result_path=self.result_path,
                                              test_lengths=self.test_lengths,
                                              rul_plot_path=self.rul_plot_path,
                                              latent_space_plot_path=self.latent_space_plot_path
                                              )
        
        print('..........Initiating Training Loop.......') 
        pbar = trange(self.n_epochs)              
        
        for epoch in pbar:

            pbar.set_description(f'Epoch {epoch}')
            training_epoch_list.append(epoch)
            
            discr_loss, trg_encoder_loss, latent_space_feat, latent_space_label, batchwise_lr = self.train_test_class.training_loop(src_loader=self.src_train_loader, 
                                                                                                                                    trg_loader=self.trg_train_loader, 
                                                                                                                                    batch_size=self.batch_size,
                                                                                                                                    epoch = epoch,
                                                                                                                                    inference = False)

            
            if all(config.latent_space_conditions):
                self.latent_space_data[epoch+1] = {'Features':latent_space_feat,
                                                   'Label': latent_space_label}
                
                    
            discr_training_loss.append(discr_loss)
            total_trg_encoder_loss.append(trg_encoder_loss)
            pbar.set_postfix(Discr_Loss = discr_loss, 
                             Trg_Encoder_Loss = trg_encoder_loss)

            
            test_pred , test_targ, test_loss = self.train_test_class.testing_loop(test_data=self.trg_test_loader)

                
            test_score,eng_scores, pred_list,label_list = utils.score_cal(test_preds=test_pred,
                                                                          test_labels=test_targ,
                                                                          test_lengths=self.test_lengths,
                                                                          inference=False)
            pbar.set_postfix(Discr_loss=discr_loss, 
                             Test_loss=test_loss,
                             Test_score=test_score)

          
            self.test_loss_list.append(test_loss)
            test_score_list.append(test_score)
            lr_list.append(batchwise_lr)
            if test_score <= min(test_score_list):
                    
                utils.model_saver(path=self.result_path,
                                  trg_model=self.trg_encoder,  
                                  discr_model=self.discriminator,
                                  score=test_score,
                                  trial=trial,
                                  loss=test_loss,
                                  epoch=epoch)
                    
                utils.accuracy_excel(test_score_list=eng_scores, 
                                     epoch=epoch, 
                                     test_loss=test_loss, 
                                     prediction_list=pred_list, 
                                     target_list=label_list, 
                                     sum_score=test_score, 
                                     result_path=self.result_path)
                    
                self.best_epoch = epoch
                self.best_loss = test_loss
                
            trial.report(test_score, epoch)
            if trial.should_prune():
                    
                raise optuna.exceptions.TrialPruned()
        
        self.best_loss_list.append(self.best_loss)
        self.best_epoch_list.append(self.best_epoch)
        
        utils.learning_rate_excel(lr_list=lr_list, 
                                  epoch_list=training_epoch_list , 
                                  loss_list=discr_training_loss,
                                  result_path=self.result_path)
        
        if all(config.latent_space_conditions):
            utils.latent_space_data_saver(latent_space_data=self.latent_space_data, 
                                          best_epoch=self.best_epoch, 
                                          path=self.result_path)
        self.plot.loss_plots(loss=discr_training_loss, train=True, name='Discriminator Loss')
        self.plot.loss_plots(loss=total_trg_encoder_loss, train=True, name='Target Encoder Loss')        
        self.plot.loss_plots(loss=self.test_loss_list, train=False, name='Testing')
        
        return min(test_score_list)
    
    def run_objective(self, n_trials, start_up_trials):
        
        sampler = optuna.samplers.TPESampler(n_startup_trials=start_up_trials, 
                                             constant_liar=True)
        
        self.study = optuna.create_study(direction='minimize',
                                         sampler=sampler,
                                         pruner=optuna.pruners.MedianPruner(n_startup_trials=config.pruner_warm_up_trial,
                                                                            n_warmup_steps=config.pruner_warm_up_step))
        self.study.optimize(self.objective, 
                            n_trials=n_trials, 
                            gc_after_trial=True)
    
    
    def create_summary(self):
        
        pruned_trials = self.study.get_trials(deepcopy=False, 
                                              states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, 
                                                states=[TrialState.COMPLETE])   
        print('\nStudy Summary')
        print(f'Number of finished trials: {len(self.study.trials):2}')
        print(f'Number of pruned trials: {len(pruned_trials):2}')
        print(f'Number of completed trials: {len(complete_trials):2}')
        self.best_trial = self.study.best_trial
        self.inference(best_trial_num=self.study.best_trial.number)
        print("Best trial:")
        print("  Value: ", self.best_trial.value)
        print("  Params: ")
        results_df = self.study.trials_dataframe()
        utils.param_results(best_trial = self.best_trial, result_excel = self.result_excel)
        utils.results_dataframe(results_df = results_df, 
                                result_excel = self.result_excel, 
                                best_epoch = self.best_epoch_list,
                                best_loss = self.best_loss_list)

    def inference(self,best_trial_num):
        trg_model, discr_model, latent_space_data= utils.inference_loader(path = pathlib.Path(self.result_path).parent, 
                                                                          n_trial=best_trial_num)
        
        best_epoch = int(self.best_trial.user_attrs['epoch']+1)
        
        self.train_test_class.trg_encoder = trg_model
        self.train_test_class.discr = discr_model
        
        test_preds , test_targ, test_loss = self.train_test_class.testing_loop(test_data = self.trg_test_loader)

        
        self.plot.testing_plots(preds = test_preds, 
                                labels = test_targ)
        
        if all(config.latent_space_conditions):

            self.plot.latent_space_plot(latent_space_data)
            self.plot.create_latent_space_gif(plot_path=self.latent_space_plot_path, 
                                              gif_path=self.gif_images_path,
                                              best_epoch = best_epoch)
            
        test_score = utils.score_cal(test_preds=test_preds,
                                     test_labels=test_targ,
                                     test_lengths=self.test_lengths,
                                     inference=True)
       
        summary = {
                    'Score':int(test_score),
                    'Test_Loss':round(test_loss,2)
                    }
        
        file_path=os.path.join(self.result_excel, 'Inference Sunmmary.txt')

        with open(file_path, 'w') as file:

            json.dump(summary,file)