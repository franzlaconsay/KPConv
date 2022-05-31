import argparse
import os
import shutil

from datasets.Pheno4d_single import Pheno4D_Single
parser = argparse.ArgumentParser()
parser.add_argument('--ply_file', type=str, required=False, default='T03_0305_a.ply')
parser.add_argument('--model', type=str, required=False, default='/home/ervin/Desktop/Thesis/O-CNN/tensorflow/script/logs/seg/pheno4d_5_k_fold_no_pre_kfold_split_3/Maize/ratio_0.01/model/iter_000020.ckpt')
parser.add_argument('--ply2points', required=False, default='ply2points')

args = parser.parse_args()

if os.path.exists('evals/'):
    shutil.rmtree('evals/')

os.makedirs('evals/')

dataset = Pheno4D_Single('Maize', args.ply_file)
dataset.check_input_pipeline_training_length()
