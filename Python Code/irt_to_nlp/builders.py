import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from irt_to_nlp.callbacks import PredictionPlotsAfterTrain,SplitDatasetWithKFoldStrategy
from irt_to_nlp.datamodule import NLPDataModule
from irt_to_nlp.config import CONFIG, Dataset
from irt_to_nlp.lit_nlp import LitNLPRegressor

def build_dataset(path_data_csv:str,dataset_name:str=CONFIG.dataset_name,
                  batch_size:int=CONFIG.batch_size,model_name:str=CONFIG.model_name):
    
    
    dataset_enum=Dataset[dataset_name]

    data_module=NLPDataModule(data_dir=os.path.join(path_data_csv,dataset_enum.value),
                              model_name=model_name,
                                            batch_size=batch_size,
                                            dataset=dataset_enum,
                                            num_workers=CONFIG.NUM_WORKERS,
                                            pin_memory=True)
    data_module.setup()
    return data_module

def get_callbacks(config,dm,only_train_and_test=False):
    #callbacks
    
    early_stopping=EarlyStopping(monitor='_val_loss',
                                 mode="min",
                                patience=10,
                                 verbose=True,
                                 check_finite =True
                                 )

    checkpoint_callback = ModelCheckpoint(
        monitor='_val_loss',
        dirpath=config.PATH_CHECKPOINT,
        filename= '-{epoch:02d}-{val_loss:.6f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    learning_rate_monitor=LearningRateMonitor(logging_interval="epoch")
    
    # save_result=True if config.num_fold==0 or only_train_and_test  else False
    save_result=True
    prediction_plot_test=PredictionPlotsAfterTrain(config.dataset_name,config.model_name,split="test",
                                                   lr_used=config.lr,save_result=save_result)
    prediction_plot_val=PredictionPlotsAfterTrain(config.dataset_name,config.model_name,split="val",
                                                  lr_used=config.lr,save_result=save_result)
    
    prediction_plot_train=PredictionPlotsAfterTrain(config.dataset_name,config.model_name,split="train",
                                                   lr_used=config.lr,save_result=save_result)
        
    callbacks=[
        # early_stopping,
        prediction_plot_val,
        prediction_plot_test,
        prediction_plot_train,
        learning_rate_monitor,
        
        
        
            ]
    if config.num_fold>=1:
        
        split_dataset=SplitDatasetWithKFoldStrategy(folds=config.num_fold,repetitions=config.repetitions,
                                                    dm=dm,
                                                    only_train_and_test=only_train_and_test)
        callbacks.append(split_dataset)
    return callbacks

def get_system(config,num_fold=0,num_repeat=0):

    return    LitNLPRegressor(lr=config.lr,
                    optim=config.optim_name,
                    model_name=config.model_name,
                    num_fold=num_fold,
                    num_repeat=num_repeat,
                    )
        

def get_trainer(wandb_logger,callbacks,config):
    
    gpus=[]
    if config.gpu0:
        gpus.append(0)
    if config.gpu1:
        gpus.append(1)
    logging.info( "gpus active",gpus)
    if len(gpus)>=2:
        distributed_backend="ddp"
        accelerator="dpp"
        plugins=DDPPlugin(find_unused_parameters=False)
    else:
        distributed_backend=None
        accelerator=None
        plugins=None

    trainer=pl.Trainer(
                    logger=wandb_logger,
                       gpus=gpus,
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,
                       log_gpu_memory=True,
                    #    distributed_backend=distributed_backend,
                    #    accelerator=accelerator,
                    #    plugins=plugins,
                       callbacks=callbacks,
                       progress_bar_refresh_rate=5,
                       
                       )
    
    return trainer
