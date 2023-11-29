import torch
import constants

DEVICE= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Config
bsc = True
batch_size = 1024
window_length= [30, 20, 32, 18]
window_step= 1
scaling_method= '-11'
window_type = constants.WindowType.Variable.value
src_dataset = constants.Subsets.FD001.value
trg_dataset = constants.Subsets.FD003.value



#Training Config 
use_gm= True
use_custom_scheduler=True
net = constants.Models.Transfomer.value
deci_mat_method = constants.MomentumGM.gtw.value
gm_method = constants.GM_Method.Momentum_GM.value
initiator = constants.MomentumGM.cold_start.value
pruner_warm_up_trial = 5
pruner_warm_up_step = 5 
random_trials = 5
num_trials = 20
n_epochs = 50

#Latent Space Config
latent_space_epochs = [1,10,20,30,40,50]
latent_space_conditions = [(('1' in src_dataset)or('2' in src_dataset)or('3' in src_dataset)),
                           (('3' in trg_dataset)or('1' in trg_dataset)) ,
                           not(('2' in src_dataset) and ('1' in trg_dataset))]
#Adaptive_MGM Config
start_epoch = 15
repeat_epoch = 5

