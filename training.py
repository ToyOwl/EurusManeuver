import timm
from train import Train, get_train_config
from models.time_distributed import TimeDistributedCollisionModel
from datasets.zürich_bicycle_time_distributed_dataset import dronet_collision_dataloaders

if __name__ == '__main__':
    setup_conf,optimizer_parameters,scheduler_parameters,loss,plotting_info, _  \
                                                             = get_train_config('config/config.yaml')

    if len(setup_conf['mean']) != len(setup_conf['std']):
        raise ValueError
    if len(setup_conf['mean']) != 1 and len(setup_conf['std']) != 3:
        raise ValueError

    model = TimeDistributedCollisionModel()

    trainer = Train(model, dronet_collision_dataloaders, setup_conf,
                    optimizer_parameters, scheduler_parameters, loss, plotting_info)
    trainer.run()