import argparse
import os
import shutil
import time
from utils.config import Config
from models.KPFCNN_model import KernelPointFCNN
from utils.tester import ModelTester
import numpy as np

from datasets.Pheno4d_single import Pheno4D_Single
parser = argparse.ArgumentParser()
parser.add_argument('--ply_file', type=str, required=False, default='T03_0305_a.ply')
parser.add_argument('--model', type=str, required=False, default='/home/ervin/Desktop/Thesis/O-CNN/tensorflow/script/logs/seg/pheno4d_5_k_fold_no_pre_kfold_split_3/Maize/ratio_0.01/model/iter_000020.ckpt')
parser.add_argument('--ply2points', required=False, default='ply2points')

args = parser.parse_args()

if os.path.exists('evals/'):
    shutil.rmtree('evals/')

os.makedirs('evals/')

def test_caller(path, step_ind, on_val, file):

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    # Load model parameters
    config = Config()
    config.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_color = 1.0
    config.validation_size = 500
    #config.batch_num = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')


    dataset = Pheno4D_Single(config.dataset.split('_')[1], file, config.input_threads)

    # Create subsample clouds of the models
    dl0 = 0.02
    dataset.load_subsampled_clouds(dl0)

    # Initialize input pipelines
    if on_val:
        dataset.init_input_pipeline(config)
    else:
        dataset.init_test_input_pipeline(config)

    #dataset.check_input_pipeline_training_length()
    
    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()
    model = KernelPointFCNN(dataset.flat_inputs, config)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    # Create a tester class
    tester = ModelTester(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ############
    # Start test
    ############

    print('Start Test')
    print('**********\n')

    tester.test_segmentation_single(model,dataset,'evals/for_evaluation.ply')
    
        
chosen_log = 'log/k0/'
chosen_snapshot = -1
on_val = False
test_caller(chosen_log, chosen_snapshot, on_val=False, file=args.ply_file)
