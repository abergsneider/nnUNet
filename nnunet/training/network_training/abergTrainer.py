#    [AB] Andres Bergsneider Modifications
'''
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2                 # [AB] Loss Function
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation # [AB] Data Augmentation
from nnunet.network_architecture.generic_UNet import Generic_UNet                               # [AB] Architecture CHECK <-------
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
'''
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2                    # [AB] Added "V2" at the end
'''
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr                                       # [AB] polyLR reducing training schedule
from batchgenerators.utilities.file_and_folder_operations import *
'''

class abergTrainer(nnUNetTrainerV2):


    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 2     # [AB] Limiting to 5