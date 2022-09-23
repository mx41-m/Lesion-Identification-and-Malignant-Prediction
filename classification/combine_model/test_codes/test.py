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
import argparse

normalize = transforms.Normalize(
    mean = [0.5, 0.5, 0.5],
    std = [0.5, 0.5, 0.5]
    )

class LesionTestDataset(Dataset):
  def __init__(self, image_path, notes_file, clinical_notes_file):
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
    self.clinical_data = np.load(clinical_notes_file, allow_pickle=True).item()

  def __len__(self):
    return len(self.test_data)

  def __getitem__(self, index):
    img_name = self.test_data[index]['image_id']
    img = Image.open(self.img_path + img_name)
    
    bbox_x1 = self.test_data[index]['bbox'][0] 
    bbox_y1 = self.test_data[index]['bbox'][1]
    bbox_x2 = bbox_x1 + self.test_data[index]['bbox'][2] 
    bbox_y2 = bbox_y1 + self.test_data[index]['bbox'][3]
    bbox_w = self.test_data[index]['bbox'][2]
    bbox_h = self.test_data[index]['bbox'][3]
    enlarge = (max(bbox_w,bbox_h) - min(bbox_w,bbox_h))/2
    if bbox_w > bbox_h:
        bbox_y1 = bbox_y1 - enlarge
        bbox_y2 = bbox_y2 + enlarge
        bbox_h = bbox_w
    else:
        bbox_x1 = bbox_x1 - enlarge
        bbox_x2 = bbox_x2 + enlarge
        bbox_w = bbox_h

    assert bbox_w == bbox_h
    box = img.crop((bbox_x1 - 0.2*bbox_w, bbox_y1 - 0.2*bbox_h, bbox_x2 + 0.2*bbox_w, bbox_y2 + 0.2*bbox_h))
    box = self.transformer(box)
    label = self.test_data[index]['category_id']
    return box, label, torch.Tensor(self.clinical_data[self.test_data[index]['image_id']])

def test(args):
  ### Dataloader
  test_dataset = LesionValDataset(args.test_image_path, args.test_image_anno, args.test_clinical_feat)
  test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8)

  # Image feature extractor
  imgfeat_model = models.resnet50(pretrained = True)
  imgfeat_model.fc = nn.Linear(in_features=2048, out_features = args.img_cls, bias=True)
  imgfeat_model.load_state_dict(torch.load(args.img_model_ck))
  imgfeat_model.cuda()
  imgfeat_model.fc = nn.Identity()
  imgfeat_model.eval()

  ### clinical linear
  clinical_combine_linear = nn.Linear(in_features = 2048 + len(test_dataset[0][-1]), out_features = args.cls_num, bias=True)
  clinical_combine_linear.load_state_dict(torch.load(args.combine_model_ck))
  clinical_combine_linear.cuda()
  clinical_combine_linear.eval()

  hit, tot = 0, 0
  score_list = []
  ground_truth_list = []
  predict = []
  all_logits = []
  for i, (data, label, clinical_data) in enumerate(test_dataloader):
      with torch.no_grad():
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True).long()
        clinical_data = clinical_data.cuda(non_blocking=True)
        image_logits = m(data)
        combine_input = torch.cat((image_logits, clinical_data),1).cuda(non_blocking=True)
        logits = clinical_combine_linear(combine_input)
        hit += (logits.topk(1, dim=1)[1] == label.view([-1,1])).sum().item()
        tot += logits.size(0)
        score_list.append(torch.softmax(logits, dim=1).cpu())
        ground_truth_list.append(label.cpu())
        predict.append(logits.topk(1, dim=1)[1].cpu().numpy())
        all_logits.append(torch.softmax(logits, dim=1).cpu().numpy())

  print('Test Accuracy: %.2f %%;' % ( hit / tot * 100))

  score_list = [item.numpy() for item in score_list]
  ground_truth_list = [item.numpy() for item in ground_truth_list]
  with open(args.test_image_anno) as f:
    anno = json.load(f)['annotations']
  
  np.save(args.res_save_path, 
    {
    'predict':predict,
    'ground_truth': ground_truth_list,
    'annotations': anno,
    'logits': all_logits,
    })


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Model utilizes both clinical information and image information')
  parser.add_argument("--test_image_path", default = './Data/Images/Private/Test/', type = str, help = 'Test images folder') 
  parser.add_argument("--test_image_anno", default = './Data/Annotations/Test/', type = str, help = 'Test images annotations')
  parser.add_argument("--test_clinical_feat", default = './Data/Clinical_Features/Test/', typr = str, help = 'Test clinical features folder')
  parser.add_argument("--img_cls", default = 2, type = int, help = 'The number of classes for image feature extractor')
  parser.add_argument("--cls_num", default = 2, type = int, help = 'The number of classes for combine model')
  parser.add_argument("--img_model_ck", default = '/Checkpoint/Image_model/', type = str, help = 'Checkpoints path for image features extractor model')
  parser.add_argument("--combine_model_ck", default = 'Checkpoint/Combine_Model/', type = str, help = 'Checkpoints path for combine model')
  parser.add_argument("--res_save_path", default = '/Results/Combine_Model/', type = str, help = 'Validation / Test results saving path for combine model')
  args = parser.parse_args()
  test(args)
