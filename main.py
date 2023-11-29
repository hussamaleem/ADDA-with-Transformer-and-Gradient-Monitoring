import optuna_class
import pathlib
import config
from datetime import datetime


main_path = pathlib.Path(__file__).parents[0]
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


runner = optuna_class.OptunaOptim(main_path=main_path, 
                                  src_dataset_num=config.src_dataset, 
                                  trg_dataset_num=config.trg_dataset,
                                  net=config.net, 
                                  window_type=config.window_type,
                                  deci_mat_method=config.deci_mat_method,
                                  gm_method=config.gm_method,
                                  initiator=config.initiator,
                                  time_stamp=time_stamp,
                                  use_gm=config.use_gm,
                                  device=config.DEVICE)

runner.run_objective(n_trials=config.num_trials, start_up_trials=config.random_trials)
runner.create_summary() 
