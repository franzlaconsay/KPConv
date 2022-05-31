import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--ply_file', type=str, required=False, default='/home/ervin/Desktop/Thesis/O-CNN/tensorflow/script/dataset/shapenet_segmentation/ply_reduced_10000/Maize/M05_0325_a.ply')
parser.add_argument('--model', type=str, required=False, default='/home/ervin/Desktop/Thesis/O-CNN/tensorflow/script/logs/seg/pheno4d_5_k_fold_no_pre_kfold_split_3/Maize/ratio_0.01/model/iter_000020.ckpt')
parser.add_argument('--ply2points', required=False, default='ply2points')

args = parser.parse_args()

ckpt = args.model
ply_file = args.ply_file
res_dir = 'evals'
ply2points = args.ply2points

if not os.path.exists(res_dir):
      os.mkdir(res_dir)

if not os.path.exists(ply_file):
      raise FileNotFoundError

def convert_to_points():
      
      filelist = open('evals/filelist.txt','w')
      filelist.write(ply_file)
      filelist.close()
      cmd = '%s --filenames evals/filelist.txt' %ply2points
      os.system(cmd)
      os.remove('evals/filelist.txt')

def ensure_ply_has_labels():
      p = open(ply_file,'r')
      lines = p.readlines()
      new_lines = []
      is_reading_header = True
      prev_line = None
      for line in lines:
            if is_reading_header:
                  if prev_line!=None and 'property float nz' in prev_line:
                        if 'element face' in line:
                              new_lines.append('property float label\n')
                  new_lines.append(line)
                  if 'end_header' in line:
                        is_reading_header = False
                  prev_line = line
            else:
                  print(line)
                  words = [x for x in line.split()]
                  print(words)
                  print(len(words))
                  if len(words) <= 6:
                        words.append('0.0')
                  new_line = ''
                  for j in range(len(words)-1):
                        new_line+=(words[j] + ' ')
                  new_line+=(words[-1]+'\n')
                  new_lines.append(new_line)
      p.close()
      p = open(ply_file,'w')
      p.writelines(new_lines)
      p.close()
      
def points_to_tfrecords():
      converter = '../util/convert_tfrecords.py'
      current_dir = os.getcwd()
      filelist = open('evals/filelist.txt','w')
      files = os.listdir('evals')
      for file in files:
            if file.endswith('.points'):
                  filename = os.path.join(current_dir, 'evals/%s' %file)
                  filelist.write(filename+' 15') #default for now
                  break
      filelist.close()
      records_name = 'evals/evalute.tfrecords'
      cmd = 'python %s --shuffle false --file_dir %s --list_file %s --records_name %s' % \
            (converter, 'evals', 'evals/filelist.txt', records_name)
      print(cmd)
      os.system(cmd)
      files = os.listdir('evals')
      for file in files:
            if not file.endswith('.tfrecords'):
                  os.remove('evals/'+file)

def run_eval():
      cmd = 'python run_seg_pheno4d_finetune.py --config configs/seg_pheno4d_eval.yaml'
      cmds = [
            cmd,
            'SOLVER.run evaluate',
            'SOLVER.gpu {},'.format(0),
            'SOLVER.ckpt {}'.format(ckpt),
            'DATA.test.location /home/ervin/Desktop/Thesis/O-CNN/tensorflow/script/evals/evalute.tfrecords',
            'MODEL.nout {}'.format(2),
            'MODEL.factor {}'.format(2),
            'LOSS.num_class {}'.format(2),
            'MODEL.name hrnet',
            ]
      cmd = ' '.join(cmds)
      print('\n', cmd, '\n')
      os.system(cmd)

def remove_tfrecords():
      files = os.listdir('evals')
      for file in files:
            if file.endswith('.tfrecords'):
                  os.remove('evals/'+file)

def save_segmented_ply():
      ply = open(ply_file,'r')
      preds = open('evals/preds.txt','r')
      evaluated = open('evals/evaluated.ply','w')
      ply_lines = ply.readlines()
      pred_lines = preds.readlines()
      idx_label = 0
      is_reading_header = True
      for i in range(len(ply_lines)):
            line = str(ply_lines[i])
            if is_reading_header:
                  evaluated.write(line)
                  if 'end_header' in line:
                        is_reading_header = False
            else:
                  words = [x for x in line.split(' ')]
                  if len(words) > 6:
                        words[-1] = str(float(pred_lines[idx_label].replace('\n','')))
                  else:
                        words.append(str(float(pred_lines[idx_label].replace('\n',''))))
                  idx_label+=1
                  for j in range(len(words)-1):
                        evaluated.write(words[j] + ' ')
                  evaluated.write(words[-1]+'\n')
      ply.close()
      preds.close()
      evaluated.close()

      #delete thats not a ply
      files = os.listdir('evals')
      for file in files:
            if not file.endswith('.ply'):
                  os.remove('evals/'+file)

ensure_ply_has_labels()
convert_to_points()
points_to_tfrecords()
run_eval()
remove_tfrecords()
save_segmented_ply()



# MODEL:
#   name: hrnet
#   channel: 4
#   nout: 15
#   depth: 6
#   factor: 2
#   signal_abs: True
#   depth_out: 6


# distort: False  # no data augmentation
#     depth: 6
#     axis: y 
#     angle: (1, 1, 1)
#     interval: (1, 1, 1)
#     scale: 0.125
#     jitter: 0.125
#     offset: 0.0
#     node_dis: True
#     location: dataset/shapenet_segmentation/datasets_reduced_10000_strict_byday/M0315_test.tfrecords
#     shuffle: 0
#     batch_size: 1
#     x_alias: data
#     return_pts: True

# cmd = ' '.join(cmds)
# print('\n', cmd, '\n')
# os.system(cmd)