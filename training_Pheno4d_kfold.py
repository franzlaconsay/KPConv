#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ShapeNetPart dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Common libs
import time
import os
import sys

# Custom libs
from utils.config import Config
from utils.trainer import ModelTrainer
from models.KPFCNN_model import KernelPointFCNN

# Dataset
from datasets.ShapeNetPart_Pheno4d import ShapeNetPartDataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


class ShapeNetPartConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name in the format 'ShapeNetPart_Object' to segment an object class independently or 'ShapeNetPart_multi'
    # to segment all objects with a single model.
    dataset = 'ShapeNetPart_Maize'

    # Number of classes in the dataset (This value is overwritten by dataset class when initiating input pipeline).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    network_model = None

    # Number of CPU threads for the input pipeline
    input_threads = 8

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # KPConv specific parameters
    num_kernel_points = 15
    first_subsampling_dl = 0.02

    # Density of neighborhoods for deformable convs (which need bigger radiuses). For normal conv we use KP_extent
    density_parameter = 5.0

    # Influence function of KPConv in ('constant', 'linear', gaussian)
    KP_influence = 'linear'
    KP_extent = 1.0

    # Aggregation function of KPConv in ('closest', 'sum')
    convolution_mode = 'sum'

    # Can the network learn modulations in addition to deformations
    modulated = False

    # Offset loss
    # 'permissive' only constrains offsets inside the big radius
    # 'fitting' helps deformed kernels to adapt to the geometry by penalizing distance to input points
    offsets_loss = 'fitting'
    offsets_decay = 0.1

    # Choice of input features
    in_features_dim = 4

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.98

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 1

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1**(1/80) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 16

    # Number of steps per epochs (cannot be None for this dataset)
    epoch_steps = None

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each snapshot
    snapshot_gap = 100

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [False, False, False]
    augment_rotation = 'none'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_occlusion = 'none'

    # Whether to use loss averaged on all points, or averaged per batch.
    batch_averaged_loss = False

    # Do we nee to save convergence
    saving = True
    saving_path = 'results_maize'


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':
    categories = ['maize', 'tomato']

    ratios = [0.01, 0.10, 0.20, 0.50, 1.00]
    muls = [10, 2, 2, 1, 1]
    max_iters = 3000

    k = 5
    kfold = KFold(k, shuffle = True, random_state = 42)
    models = [1, 2, 3, 4, 5, 6, 7]

    for cat in categories:
        # training ratios
        for ratio_i in range(len(ratios)):
            max_epochs = max_iters * ratios[ratio_i] * muls[ratio_i]

            # k-folds
            for k_i in enumerate(kfold.split(models)):
                print('==================================================')
                print('category: %s, ratio: %s, kfold: %s' % (cat, str(ratios[ratio_i]), str(k_i[0])))
                print('==================================================')
                k_fold = k_i[0]
                train_indices = k_i[1][0]
                test_indices = k_i[1][1]

                TRAIN_SAMPLE = ratios[ratio_i]
                MAX_EPOCH = int(max_epochs)
                CATEGORY = cat
                LOG_DIR = os.path.join('log', cat, str(int(TRAIN_SAMPLE*100)), 'k'+str(k_fold))
                DATASET = 'ShapeNetPart_Maize'

                if CATEGORY == 'maize':
                    DATASET = 'ShapeNetPart_Maize'
                elif CATEGORY == 'tomato':
                    DATASET = 'ShapeNetPart_Tomato'
                else:
                    pass

                ##########################
                # Initiate the environment
                ##########################

                # Choose which gpu to use
                GPU_ID = '0'

                # Set GPU visible device
                os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

                # Enable/Disable warnings (set level to '0'/'3')
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

                ###########################
                # Load the model parameters
                ###########################

                config = ShapeNetPartConfig()
                config.dataset = DATASET
                config.num_classes = None
                config.network_model = None
                config.input_threads = 8
                config.architecture = ['simple',
                                      'resnetb',
                                      'resnetb_strided',
                                      'resnetb',
                                      'resnetb_strided',
                                      'resnetb_deformable',
                                      'resnetb_deformable_strided',
                                      'resnetb_deformable',
                                      'resnetb_deformable_strided',
                                      'resnetb_deformable',
                                      'nearest_upsample',
                                      'unary',
                                      'nearest_upsample',
                                      'unary',
                                      'nearest_upsample',
                                      'unary',
                                      'nearest_upsample',
                                      'unary']
                config.num_kernel_points = 15
                config.first_subsampling_dl = 0.02
                config.density_parameter = 5.0
                config.KP_influence = 'linear'
                config.KP_extent = 1.0
                config.convolution_mode = 'sum'
                config.modulated = False
                config.offsets_loss = 'fitting'
                config.offsets_decay = 0.1
                config.in_features_dim = 4
                config.use_batch_norm = True
                config.batch_norm_momentum = 0.98
                config.max_epoch = MAX_EPOCH
                config.learning_rate = 1e-2
                config.momentum = 0.98
                config.lr_decays = {i: 0.1**(1/80) for i in range(1, MAX_EPOCH)}
                config.grad_clip_norm = 100.0
                config.batch_num = 4
                config.epoch_steps = None
                config.validation_size = 50
                config.snapshot_gap = 100
                config.augment_scale_anisotropic = True
                config.augment_symmetries = [False, False, False]
                config.augment_rotation = 'none'
                config.augment_scale_min = 0.9
                config.augment_scale_max = 1.1
                config.augment_noise = 0.001
                config.augment_occlusion = 'none'
                config.batch_averaged_loss = False
                config.saving = True
                config.saving_path = LOG_DIR

                ##############
                # Prepare Data
                ##############

                print()
                print('Dataset Preparation')
                print('*******************')

                # Initiate dataset configuration
                dataset = ShapeNetPartDataset(config.dataset.split('_')[1], config.input_threads, train_indices, test_indices, TRAIN_SAMPLE)

                # Create subsampled input clouds
                dl0 = config.first_subsampling_dl
                dataset.load_subsampled_clouds(dl0)

                # Initialize input pipelines
                dataset.init_input_pipeline(config)

                # Test the input pipeline alone with this debug function
                # dataset.check_input_pipeline_timing(config)

                ##############
                # Define Model
                ##############

                print('Creating Model')
                print('**************\n')
                t1 = time.time()

                # Model class
                model = KernelPointFCNN(dataset.flat_inputs, config)

                # Trainer class
                trainer = ModelTrainer(model)
                t2 = time.time()

                print('\n----------------')
                print('Done in {:.1f} s'.format(t2 - t1))
                print('----------------\n')

                ################
                # Start training
                ################

                print('Start Training')
                print('**************\n')

                trainer.train(model, dataset)
