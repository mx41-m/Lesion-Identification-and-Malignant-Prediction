#!/usr/bin/env python

from PIL import Image, ExifTags
import numpy as np
import os, sys
import json
import urllib.request
import io

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

normalize = transforms.Normalize(
    mean = [0.5, 0.5, 0.5],
    std = [0.5, 0.5, 0.5]
    )

class LesionTestDataset(Dataset):
  def __init__(self, image_path, notes_file):
    self.img_path = image_path
    with open(notes_file) as f:
        notes = f.read()
    notes = json.loads(notes)
    self.test_data = notes['annotations']
    self.transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
     ])

  def __len__(self):
    return len(self.test_data)

  def __getitem__(self, index):
    img_name = self.test_data[index]['image_id']
    try:
      img_raw = urllib.request.urlopen('http://127.0.0.1:5000/' + img_name).read()
      img_raw = io.BytesIO(img_raw)
      img = Image.open(img_raw)
    except:
      img = Image.open(self.img_path + img_name)

    img = self.transformer(img)
    label = self.test_data[index]['category_id']

    return img, label

def test(args)
  # prepare DataSet
  test_dataloader = DataLoader(LesionValDataset(args.test_image_path, args.test_image_anno), batch_size = args.batch_size, shuffle=False, num_workers=8)

  # Define Model
  m = models.resnet50(pretrained = True)
  m.fc = nn.Linear(in_features=2048, out_features=args.cls_num, bias=True)
  nn.init.kaiming_normal_(m.fc.weight)
  nn.init.constant_(m.fc.bias, 0.)
  
  m.cuda()
  m.load_state_dict(torch.load(args.img_model_ck))

  m.eval()
  hit, tot = 0, 0
  score_list = []
  ground_truth_list = []
  predict = []
  all_logits = []
  for i, (data, label) in enumerate(test_loader):
    with torch.no_grad():
      data = data.cuda(non_blocking=True)
      label = label.cuda(non_blocking=True).long()
      logits = F.softmax(m(data), dim=1)
      hit += (logits.topk(1, dim=1)[1] == label.view([-1,1])).sum().item()
      tot += logits.size(0)
      score_list.append(torch.softmax(logits, dim=1).cpu())
      ground_truth_list.append(label.cpu())
      predict.append(logits.topk(1, dim=1)[1].cpu().numpy())
      all_logits.append(logits.cpu().numpy())

  total_ground_truth_list = torch.cat(ground_truth_list, 0)
  total_score_list = torch.cat(score_list, 0)

  print('Test Accuracy: %.2f %%' % (hit / tot * 100))

  score_list = [item.numpy() for item in score_list]
  ground_truth_list = [item.numpy() for item in ground_truth_list]
  with open(args.test_image_anno) as f:
    anno = json.load(f)['annotations']

  np.save(args.res_save_path, {
    'predict':predict,
    'ground_truth':ground_truth_list,
    'annotations': anno,
    'logits': all_logits,
    })

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Model utilizes both clinical information and image information')
  parser.add_argument("--test_image_path", default = './Data/Images/Private/Test/', type = str, help = 'Test images folder')
  parser.add_argument("--test_image_anno", default = './Data/Annotations/Test/', type = str, help = 'Test images annotations')
  parser.add_argument("--cls_num", default = 2, type = int, help = 'The number of classes for classification model')
  parser.add_argument("--img_model_ck", default = 'Checkpoint/ISIC_augmentation_Model/Entire_Image', type = str, help = 'Checkpoints path for ISIC augmentation model')
  parser.add_argument("--res_save_path", default = '/Results/ISIC_augmentation_Model/', type = str, help = 'Validation / Test results saving path for combine model')
  args = parser.parse_args()
  test(args)

