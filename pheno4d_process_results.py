import os
import json
import xlsxwriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
LOG_DIR = 'log'

categories = ['maize', 'tomato']

for cat in categories:
  cat_folder = os.path.join(BASE_DIR, LOG_DIR, cat)
  ratio_folders = os.listdir(cat_folder)
  for r_folder in ratio_folders:
    k_folders = os.listdir(os.path.join(cat_folder, r_folder))
    results_dir = os.path.join(BASE_DIR, 'results', cat, r_folder)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    #totals
    total_stats = {}
    total_stats['accuracy'] = 0
    total_stats['miou'] = 0
    total_stats['stem_iou'] = 0
    total_stats['leaf_iou'] = 0
    total_stats['stem_recall'] = 0
    total_stats['leaf_recall'] = 0
    total_stats['stem_precision'] = 0
    total_stats['leaf_precision'] = 0
    total_stats['tn'] = 0
    total_stats['fp'] = 0
    total_stats['fn'] = 0
    total_stats['tp'] = 0
    for k_folder in k_folders:
      epochs_path = os.path.join(cat_folder, r_folder, k_folder, 'epochs.json')
      with open(epochs_path, 'r') as f:
        data = json.load(f)
        largest_miou = 0
        largest_miou_i = 0
        # find index of largest mIoU for k_i
        for i in range(len(data['epochs'])):
          cur_miou = data['epochs'][i]['miou']
          if cur_miou > largest_miou:
            largest_miou = cur_miou
            largest_miou_i = i
        for j in data['epochs'][largest_miou_i]:
          total_stats[j] += data['epochs'][largest_miou_i][j]
    # writing output
    workbook = xlsxwriter.Workbook(os.path.join(results_dir, '%s_%s.xlsx' % (cat, r_folder)))
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for k in total_stats:
      total_stats[k] /= len(k_folders) # average
      worksheet.write(row, col, k)
      try:
        worksheet.write(row+1, col, total_stats[k])
      except:
        worksheet.write(row+1, col, 'nan')
      col += 1
    workbook.close()