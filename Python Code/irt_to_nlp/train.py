import datetime
import logging
import sys

from pytorch_lightning.core import datamodule
from torch.utils import data
#cd /home/dcast/adversarial_project ; /usr/bin/env /home/dcast/anaconda3/envs/deep_learning_torch/bin/python -- /home/dcast/adversarial_project/irt_to_nlp/train.py 
# sys.path.append("/content/adversarial_project") #to work in colab
sys.path.append("/home/dcast/adversarial_project")
import os
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from openml.autotune import autotune_lr
from irt_to_nlp.builders import build_dataset, get_callbacks, get_trainer,get_system
from irt_to_nlp.config import CONFIG, create_config_dict

import pandas as pd

import uuid

    


def apply_train_test():
    def get_new_system_and_do_training_and_test(config,data_module,wandb_logger,callbacks,num_repeat=None,num_fold=None,run_test:bool=False):
        model=get_system(config,num_fold,num_repeat=num_repeat)
        
        # test_dataloader=data_module.test_dataloader()
        #create trainer
        callbacks[0].best_score=torch.tensor(np.Inf)
        trainer=get_trainer(wandb_logger,callbacks,config)
        
        
        result=trainer.fit(model,data_module)
        if run_test:
            result=trainer.test(model,test_dataloaders=data_module.test_dataloader())
        
            return result
        return result
    
    # def generate_csv_results(results:list,data_name:str):
    #     df=pd.DataFrame(results)
    #     df.to_csv(f"{data_name}.csv")
        
    init_id=uuid.uuid1().int  
    config=CONFIG()
    config_dict=create_config_dict(config)
    config_dict["id_group"]=init_id
    wandb.init(
        project='IRT-project-NLP',
                entity='dcastf01',
                config=config_dict)
    
    wandb_logger = WandbLogger( 
                    # offline=True,
                    log_model=False
                    )
    
    config =wandb.config
    
    data_module=build_dataset(path_data_csv=config.path_data,
                              dataset_name=config.dataset_name,
                              batch_size=config.batch_size,
                              model_name=config.model_name
                              )
    callbacks=get_callbacks(config,data_module)
    num_fold=config.num_fold
    
    if num_fold is None or num_fold==0:
        wandb.run.name=config.model_name[:5]+" "+\
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %X")
        get_new_system_and_do_training_and_test(config,data_module,wandb_logger,callbacks,num_fold=num_fold,run_test=True)
        
    else:
        results=[]
        for num_repeat in range(config.repetitions):
            for fold in range(num_fold):
                print(f"Repeticion {num_repeat}, fold {fold}")
                if not num_repeat==0 or not fold==0:
                    config=CONFIG()
                    config_dict=create_config_dict(config)
                    config_dict["id_group"]=init_id
                    wandb.init(
                        project='IRT-project-NLP',
                                entity='dcastf01',
                                config=config_dict)
                    
                    wandb_logger = WandbLogger( 
                                    # offline=True,
                                    log_model=False
                                    )
                    
                    config =wandb.config
                config=wandb.config
                wandb.run.name=str(num_repeat)+"_"+str(fold)+" "+config.model_name[:5]+" "+\
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %X")
                # wandb.run.id_group=init_id
                
                result=get_new_system_and_do_training_and_test(config,data_module,
                                                               wandb_logger,callbacks,
                                                               num_repeat=num_repeat,
                                                               num_fold=fold,
                                                               run_test=False)
                if results:
                    results.append(*result)
                wandb.finish()
        # config=CONFIG()
        # config_dict=create_config_dict(config)
       
        # wandb.init(
        #     project='IRT-project-NLP',
        #             entity='dcastf01',
        #             config=config_dict)
        
        # wandb_logger = WandbLogger( 
        #                 # offline=True,
        #                 log_model=False
        #                 )
        
        # config =wandb.config
        # config=wandb.config
        # wandb.run.name="final"+config.model_name[:5]+" "+\
        #     datetime.datetime.utcnow().strftime("%Y-%m-%d %X")
        # # wandb.run.id_group=init_id
        # callbacks=get_callbacks(config,data_module,only_train_and_test=True)
        # result=get_new_system_and_do_training_and_test(config,data_module,
        #                                                wandb_logger,
        #                                                callbacks,num_fold=num_fold,
        #                                                run_test=True
        #                                                )
        # results.append(*result)
        
        # generate_csv_results(results,config.dataset_name)
            # "construir lo del fold"
        
            
def main():
    os.environ["WANDB_IGNORE_GLOBS"]="*.ckpt"
    torch.manual_seed(0)
    print("empezando setup del experimento")
    torch.backends.cudnn.benchmark = True
    
    #aplicar todo lo del fold a partir de aquí
    
    
    apply_train_test()
if __name__ == "__main__":
    
    main()
