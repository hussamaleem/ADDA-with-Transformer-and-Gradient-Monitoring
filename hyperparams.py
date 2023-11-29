class Hyperparameters():
    def __init__(self, 
                 trial = None,
                 use_gm = False,
                 gm_method = None,
                 use_custom_scheduler = None
                 ):
        
        self.trial = trial
        self.use_gm = use_gm
        self.gm_method = gm_method 
        self.use_custom_scheduler = use_custom_scheduler
    
    def custom_scheduler(self):
        
        if self.use_custom_scheduler:
            warmup_ratio = self.trial.suggest_float("LR_Warmup_Ratio", 
                                                    low = 0.4, 
                                                    high = 0.75, 
                                                    step = 0.05)
            trg_encoder_lr = 0
            
        else:
            
            warmup_ratio = 0
            trg_encoder_lr = self.trial.suggest_float("Trg_Encoder_LR",
                                                      low=0.001,
                                                      high=0.01,
                                                      log=True)
        
        scheduler_params = {'Warm_Up_Ratio': warmup_ratio,
                            'Trg_Encoder_LR': trg_encoder_lr}
        
        
        return scheduler_params
    
    def adda_training(self):
        
        num_epochs = 75
        discr_lr = self.trial.suggest_categorical('lr', [0.01,0.001,0.0001])
        
        training_params = {"Discr_Learning_Rate": discr_lr,
                           'Num_Epochs': num_epochs}
        
        return training_params
        
    def Momentum_GM_training(self):
        
        num_epochs = 50
        
        discr_lr = self.trial.suggest_float("Learning_Rate",
                                            low = 1e-4,
                                            high = 1e-2,
                                            log = True)
        
        learn_factor = self.trial.suggest_float('Learning_Factor',
                                             low = 0.6,
                                             high = 0.9,
                                             log = True)
        
        momentum =  self.trial.suggest_float('Momentum',
                                             low = 0.75,
                                             high = 0.99,
                                             log = True)
        
        training_params = {"Discr_Learning_Rate": discr_lr,
                           'Learning_Factor': learn_factor,
                           'Momentum': momentum,
                           'Num_Epochs': num_epochs}
        
        return training_params
    
    def Adaptive_MGM_training(self):
        
        num_epochs = 50
        
        
        discr_lr = self.trial.suggest_float("Learning_Rate",
                                            low = 1e-4,
                                            high = 1e-2,
                                            log = True)
        
        alpha = self.trial.suggest_float('Alpha', 
                                         low = 0.01,
                                         high = 0.1,
                                         log = True)
        
        learn_factor = self.trial.suggest_float('Learning_Factor',
                                             low = 0.6,
                                             high = 0.9,
                                             log = True)
        
        momentum =  self.trial.suggest_float('Momentum',
                                             low = 0.75,
                                             high = 0.99,
                                             log = True)
        
        training_params = {"Discr_Learning_Rate": discr_lr,
                           'Learning_Factor': learn_factor,
                           'Momentum': momentum,
                           'Num_Epochs': num_epochs,
                           'Alpha': alpha}
        
        return training_params
    def Vanilla_GM_training(self):
    
        num_epochs = 40
        start_epoch = 10
        repeat_epoch = 2
        
        discr_lr = self.trial.suggest_float("Learning_Rate",
                                                 low = 1e-5,
                                                 high = 1e-2,
                                                 log = True)
    
        learn_factor = self.trial.suggest_float('Learning_Factor', 
                                                     low = 0.5,
                                                     high = 0.8,
                                                     log = True)
    

    
        training_params = {"Discr_Learning_Rate": discr_lr,
                           'Learning_Factor': learn_factor,
                           'Start_Epoch': start_epoch,
                           'Repeat_Epoch': repeat_epoch,
                           'Num_Epochs': num_epochs}
    
        return training_params
        
    def discriminator(self):
        
        if self.use_gm:
            hidden_size = self.trial.suggest_int('Hidden_Size', 
                                                 low = 48,
                                                 high = 160, 
                                                 step = 16)
        else:
            hidden_size =  self.trial.suggest_categorical('Hidden_Size', [32,64,128])
            
        return {"Hidden_Size": hidden_size}
    
    def get_hyperparams(self):
        
        discriminator_params = self.discriminator()
        scheduler_params = self.custom_scheduler()
        if self.use_gm:    
            training_params = getattr(self, f'{self.gm_method}_training')()
        else:
            training_params = self.adda_training()
            
        return {**discriminator_params, **training_params, **scheduler_params}