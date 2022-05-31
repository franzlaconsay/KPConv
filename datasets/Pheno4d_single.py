# Basic libs
from this import d
import tensorflow as tf
import numpy as np
import time
import pickle
import json
import os

# PLY reader
from utils.ply import read_ply, write_ply

# OS functions
from os import listdir, makedirs
from os.path import exists, join

# Dataset parent class
from datasets.common import Dataset

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

from sklearn.model_selection import train_test_split
from random import sample
import math


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


class Pheno4D_Single(Dataset):
    def __init__(self,class_name, file , input_threads=8):
        Dataset.__init__(self, 'ShapeNetPart_' + class_name)
        self.category = class_name.upper()
        self.filename = file
        if self.category == 'MAIZE':
          self.label_to_names = {0: 'Maize'}
        elif self.category == 'TOMATO':
          self.label_to_names = {0: 'Tomato'}

        # List of classes ignored during training (can be empty)
        #self.ignored_labels = np.array([])

        # Number of parts for each object
        if self.category == 'MAIZE' or self.category == 'TOMATO':
          self.num_parts = [2]
        
        self.ShapeNetPartType = class_name
        self.network_model = 'segmentation'
        self.num_train = None
        self.num_test = None

        self.path = 'evals/'
        self.num_threads = input_threads

        self.prepare_ShapeNetPart_ply()

        return
    
    def prepare_ShapeNetPart_ply(self):
      # Get filename

      points,labels = self.get_points_and_labels_from_ply(self.filename)
      # Center and rescale point for 1m radius
      # test later if there is no need to do this because this will result in smaller point cloud
      
      pmin = np.min(points, axis=0)
      pmax = np.max(points, axis=0)
      points -= (pmin + pmax) / 2
      scale = np.max(np.linalg.norm(points, axis=1))
      points *= 1.0 / scale

      print("Points: ", points[:5])
      print("Labels: ", labels[:5])

      ply_name = 'evals/for_evaluation.ply'
      # Switch y and z dimensions
      #points = points[:, [0, 2, 1]]

      # Save in ply format
      write_ply(ply_name, (points, labels), ['x', 'y', 'z', 'label'])


    def get_points_and_labels_from_ply(self,ply_file):
      p = open(ply_file,'r')
      points = []
      labels = []
      lines = p.readlines()
      is_reading_header = True
      for line in lines:
            if is_reading_header:
                  if 'end_header' in line:
                        is_reading_header = False
            else:
                  words = [x for x in line.split()]
                  if len(words) <= 6:
                    points.append([float(x) for x in words[:3]])
                    labels.append(0.0)
                  else:
                    points.append([float(x) for x in words[:3]])
                    labels.append(float(words[-1]))
      p.close()
      return (np.array(points),np.array(labels)+1)
  
    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory
        """

        # if 0 < subsampling_parameter <= 0.01:
        #     raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_points = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_point_labels = {'training': [], 'validation': [], 'test': []}

        # Load wanted points if possible
        print('\nLoading training points')
        filename = join(self.path, 'eval_train_{:.3f}_record.pkl'.format(subsampling_parameter))

        if exists(filename):
            with open(filename, 'rb') as file:
                self.input_labels['training'], \
                self.input_points['training'], \
                self.input_point_labels['training'] = pickle.load(file)
        
        #store file in training
        #presumed to have no effect but just following to avoid errors
        
        else:
            data = read_ply('evals/for_evaluation.ply')
            points = np.vstack((data['x'], data['y'], data['z'])).T
            print(points)
            point_labels = data['label']
            points = points.astype(np.float32)
            print("Point Type", points.dtype)
            point_labels = point_labels.astype(np.int32)
            print("Label Type", point_labels.dtype)
            if subsampling_parameter > 0:
                sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                            sampleDl=subsampling_parameter)
                self.input_points['training'] += [sub_points]
                self.input_point_labels['training'] += [sub_labels]
            else:
                self.input_points['training'] += [points]
                self.input_point_labels['training'] += [point_labels]

            # just make everything Maize for now
            # again no effect
            self.input_labels['training'] = np.array([0])

            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_labels['training'],
                                self.input_points['training'],
                                self.input_point_labels['training']), file)
            
        # Test files
        # this will have an effect
        
        filename = join(self.path, 'eval_test_{:.3f}_record.pkl'.format(subsampling_parameter))
        if exists(filename):
            with open(filename, 'rb') as file:
                self.input_labels['test'], \
                self.input_points['test'], \
                self.input_point_labels['test'] = pickle.load(file)

        # Else compute them from original points
        else:
            data = read_ply('evals/for_evaluation.ply')
            points = np.vstack((data['x'], data['y'], data['z'])).T
            point_labels = data['label']
            points = points.astype(np.float32)
            print("Point Type", points.dtype)
            point_labels = point_labels.astype(np.int32)
            print("Label Type", point_labels.dtype)
            if subsampling_parameter > 0:
                sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                            sampleDl=subsampling_parameter)
                self.input_points['test'] += [sub_points]
                self.input_point_labels['test'] += [sub_labels]
            else:
                self.input_points['test'] += [points]
                self.input_point_labels['test'] += [point_labels]

            # just make everything Maize for now
            # no effect
            self.input_labels['test'] = np.array([0])

            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_labels['test'],
                                self.input_points['test'],
                                self.input_point_labels['test']), file)
            
        # Change to 0-based labels
        self.input_point_labels['training'] = [p_l - 1 for p_l in self.input_point_labels['training']]
        self.input_point_labels['test'] = [p_l - 1 for p_l in self.input_point_labels['test']]

        # Test = validation
        self.input_labels['validation'] = self.input_labels['test']
        self.input_points['validation'] = self.input_points['test']
        self.input_point_labels['validation'] = self.input_point_labels['test']
        
        return
    
    def get_batch_gen(self, split, config):

        ################
        # Def generators
        ################

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}

        # Reset potentials
        self.potentials[split] = np.random.rand(len(self.input_labels[split])) * 1e-3

        def variable_batch_gen_multi():

            # Initiate concatanation lists
            tp_list = []
            tl_list = []
            tpl_list = []
            ti_list = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'training':
                gen_indices = np.random.permutation(self.num_train)

            elif split == 'validation':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            elif split == 'test':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            else:
                raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

            # Generator loop
            for i, rand_i in enumerate(gen_indices):

                # Get points
                new_points = self.input_points[split][rand_i].astype(np.float32)
                n = new_points.shape[0]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit:
                    yield (np.concatenate(tp_list, axis=0),
                           np.array(tl_list, dtype=np.int32),
                           np.concatenate(tpl_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tl_list = []
                    tpl_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [new_points]
                tl_list += [self.input_labels[split][rand_i]]
                tpl_list += [np.squeeze(self.input_point_labels[split][rand_i])]
                ti_list += [rand_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tp_list, axis=0),
                   np.array(tl_list, dtype=np.int32),
                   np.concatenate(tpl_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        def variable_batch_gen_segment():

            # Initiate concatanation lists
            tp_list = []
            tpl_list = []
            ti_list = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'training':
                gen_indices = np.random.permutation(self.num_train)
            elif split == 'validation':
                gen_indices = np.random.permutation(self.num_test)
            elif split == 'test':
                gen_indices = np.arange(self.num_test)
            else:
                raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

            # Generator loop
            for i, rand_i in enumerate(gen_indices):

                # Get points
                new_points = self.input_points[split][rand_i].astype(np.float32)
                n = new_points.shape[0]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit:
                    yield (np.concatenate(tp_list, axis=0),
                           np.concatenate(tpl_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tpl_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [new_points]
                tpl_list += [np.squeeze(self.input_point_labels[split][rand_i])]
                ti_list += [rand_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tp_list, axis=0),
                   np.concatenate(tpl_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        ###################
        # Choose generators
        ###################

        if self.ShapeNetPartType == 'multi':
            # Generator types and shapes
            gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
            gen_shapes = ([None, 3], [None], [None], [None], [None])
            return variable_batch_gen_multi, gen_types, gen_shapes

        elif self.ShapeNetPartType in self.label_names:

            # Generator types and shapes
            gen_types = (tf.float32, tf.int32, tf.int32, tf.int32)
            gen_shapes = ([None, 3], [None], [None], [None])
            return variable_batch_gen_segment, gen_types, gen_shapes
        else:
            raise ValueError('Unsupported ShapeNetPart dataset type')

    def get_tf_mapping(self, config):

        def tf_map_multi(stacked_points, object_labels, point_labels, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param labels: Tensor with size [None] where None is the number of batch
            :param stack_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, tf.square(stacked_points)), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds,
                                                     object_labels=object_labels)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        def tf_map_segment(stacked_points, point_labels, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param labels: Tensor with size [None] where None is the number of batch
            :param stack_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, tf.square(stacked_points)), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        if self.ShapeNetPartType == 'multi':
            return tf_map_multi

        elif self.ShapeNetPartType in self.label_names:
            return tf_map_segment
        else:
            raise ValueError('Unsupported ShapeNetPart dataset type')


    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def check_input_pipeline_timing(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                t += [time.time()]

                # Average timing
                mean_dt = 0.99 * mean_dt + 0.01 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display
                if (t[-1] - last_display) > 1.0 / 10:
                    last_display = t[-1]
                    message = 'Step {:08d} : timings {:4.2f} {:4.2f} - {:d} x {:d}'
                    print(message.format(training_step,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         neighbors[0].shape[0],
                                         neighbors[0].shape[1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return
    def check_input_pipeline_training_length(self):
        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        d_len = 0
        try:
            # Run one step of the model.
            # Get next inputs
            np_flat_inputs_1 = self.sess.run(self.flat_inputs)
            d_len+=1
        except tf.errors.OutOfRangeError:
            print('End of train dataset')
            print("Length of train dataset: ", d_len)
            
            self.sess.run(self.train_init_op)

        return
    
    def check_input_pipeline_2(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        epoch = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]

                # Get next inputs
                np_flat_inputs_1 = self.sess.run(self.flat_inputs)


            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1
        return

    def check_debug_input(self, config, path):

        # Get debug file
        file = join(path, 'all_debug_inputs.pkl')
        with open(file, 'rb') as f1:
            inputs = pickle.load(f1)

        # Print inputs
        nl = config.num_layers
        for layer in range(nl):
            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2 * nl + layer]
            upsamples = inputs[3 * nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / (np.prod(pools.shape) + 1e-6)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / (np.prod(upsamples.shape) + 1e-6)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        ind += 1
        if config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        # Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2 * nl + layer]
            upsamples = inputs[3 * nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))

            if np.prod(upsamples.shape) > 0:
                max_n = np.max(upsamples)
                nums = np.sum(upsamples < max_n - 0.5, axis=-1)
                print('min upsamples =>', np.min(nums))

        print('\nFinished\n\n')
        time.sleep(0.5)

        self.flat_inputs = [tf.Variable(in_np, trainable=False) for in_np in inputs]

