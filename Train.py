import Datasets
import Models
import Trainer
import torch
from functools import partial
import numpy as np
import random

if __name__ == '__main__':
    # torch.manual_seed(12)
    # np.random.seed(12)
    # random.seed(12)

    load_size = (500,400)
    input_size = (384,384)
    EPOCHS = 30

    label_dist = [2745, 2050,  415, 3660, 4085,  545, 549, 410, 83, 732, 817,
        109, 549, 410, 83, 732, 817, 109]

    label_dist = [1647, 1230, 830, 1464, 1634, 1090, 549, 410, 166, 732, 817,
     218, 549, 410, 166, 732, 817, 218] # 51th

    total_cnt = sum(label_dist)
    gamma = 2
    label_dist = [ (1-(l/total_cnt))**gamma for l in label_dist ]
    weight= torch.FloatTensor(label_dist).to('cuda')


    dataset = Datasets.OverSamplingMaskDataset

    transform = Datasets.albumentation(size=input_size,
                                       use_randcrop=True, use_center_crop=False, use_randomreisze_crop=False,
                                       use_filp=True, use_rotate=True, use_blur=False,
                                       use_noise=False, use_normalize=True, use_CLAHE=True,
                                       use_invert=False, use_equalize=False, use_posterize=True,
                                       use_soloarize=False, ues_jitter=False, use_Brightness=False,
                                       use_Gamma=False, use_brightcontrast=False, use_cutout=False,
                                       use_totensor=True
                                       )

    # transform = Datasets.albumentation(size=input_size,
    #                                    use_randcrop=True, use_center_crop=False, use_randomreisze_crop=False,
    #                                    use_filp=True, use_rotate=True, use_blur=False,
    #                                    use_noise=False, use_normalize=True, use_CLAHE=False,
    #                                    use_invert=False, use_equalize=False, use_posterize=True,
    #                                    use_soloarize=False, ues_jitter=False, use_Brightness=False,
    #                                    use_Gamma=False, use_brightcontrast=False, use_cutout=False,
    #                                    )

    scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=1, eta_min=0)

    CONFIG = {
        "model" : Models.ResNext,
        "model_name" : 'ResnNext',
        "nick_name" : '384384_scheduler',
        "transform" : transform,
        "dataset" : dataset,
        "learning_rate" : 1e-05,
        "weight_decay" : 5e-02,
        "train_ratio" : 0.8,
        "batch_size" : 32,
        "epoch" : EPOCHS,
        "optimizer" : torch.optim.Adam,
        "loss_function" : torch.nn.CrossEntropyLoss(),
        "scheduler" : scheduler,
        "input_size" : input_size,
        "load_size" : load_size
    }

    Trainer.Trainer(**CONFIG).train()
    # Trainer.Trainer(**CONFIG)




    # dataset = Datasets.AlbumentationMaskDataset
    # transform = Datasets.get_augmentation(size=input_size, use_flip=True,
    #                              use_color_jitter=False, use_gray_scale=False, use_normalize=True)
    #
    # transform = Datasets.albumentation(size=input_size, use_filp=True, use_rotate=True, use_blur=True,
    #                                    use_noise=True, use_normalize=True, use_CLAHE=False,
    #                                    use_invert=False, use_equalize=False, use_posterize=False,
    #                                    use_soloarize=False, ues_jitter=False, use_Brightness=True,
    #                                    use_Gamma=True, use_brightcontrast=True, use_cutout=True,
    #                                    )