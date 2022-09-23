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

normalize = transforms.Normalize(
    mean = [0.5, 0.5, 0.5],
    std = [0.5, 0.5, 0.5]
    )

class LesionTrainDataset(Dataset):
  def __init__(self, image_path, notes_file):
    self.img_path = image_path
    with open(notes_file) as f:
        notes = f.read()
    notes = json.loads(notes)
    self.train_data = notes['annotations'] #include image_id, bbox, category_id
    self.transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        normalize
    ])

  def __len__(self):
    return len(self.train_data)

  def __getitem__(self, index):
    img_name = self.train_data[index]['image_id']
    try:
      img_raw = urllib.request.urlopen('http://127.0.0.1:5001/' + img_name).read()
      img_raw = io.BytesIO(img_raw)
      img = Image.open(img_raw)
    except:
      img = Image.open(self.img_path + img_name)

    bbox_x1 = self.train_data[index]['bbox'][0]
    bbox_y1 = self.train_data[index]['bbox'][1]
    bbox_x2 = bbox_x1 + self.train_data[index]['bbox'][2]
    bbox_y2 = bbox_y1 + self.train_data[index]['bbox'][3]
    bbox_w = self.train_data[index]['bbox'][2]
    bbox_h = self.train_data[index]['bbox'][3]
    
    ### Enlarge the bounding box for better input since the bounding boxes may not perfect
    enlarge = (max(bbox_w, bbox_h) - min(bbox_w, bbox_h)) / 2
    if bbox_w > bbox_h:
        bbox_y1 = bbox_y1 - enlarge
        bbox_y2 = bbox_y2 + enlarge
        bbox_h = bbox_w
    else:
        bbox_x1 = bbox_x1 - enlarge
        bbox_x2 = bbox_x2 + enlarge
        bbox_w = bbox_h
    
    assert bbox_w == bbox_h
    img = img.crop((bbox_x1 - 0.2*bbox_w, bbox_y1 - 0.2*bbox_h, bbox_x2 + 0.2*bbox_w, bbox_y2 + 0.2*bbox_h))
    
    img = self.transformer(img) 
    label =  self.train_data[index]['category_id']
    return img, label

class LesionValDataset(Dataset):
  def __init__(self, image_path, notes_file):
    self.img_path = image_path
    with open(notes_file) as f:
        notes = f.read()
    notes = json.loads(notes)
    self.val_data = notes['annotations']
    self.transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
     ])

  def __len__(self):
    return len(self.val_data)

  def __getitem__(self, index):
    img_name = self.train_data[index]['image_id']
    try:
      img_raw = urllib.request.urlopen('http://127.0.0.1:5001/' + img_name).read()
      img_raw = io.BytesIO(img_raw)
      img = Image.open(img_raw)
    except:
      img = Image.open(self.img_path + img_name)
    
    bbox_x1 = self.train_data[index]['bbox'][0]
    bbox_y1 = self.train_data[index]['bbox'][1]
    bbox_x2 = bbox_x1 + self.train_data[index]['bbox'][2]
    bbox_y2 = bbox_y1 + self.train_data[index]['bbox'][3]
    bbox_w = self.train_data[index]['bbox'][2]
    bbox_h = self.train_data[index]['bbox'][3]
    
    ### Enlarge the bounding box for better input since the bounding boxes may not perfect
    enlarge = (max(bbox_w, bbox_h) - min(bbox_w, bbox_h)) / 2
    if bbox_w > bbox_h:
        bbox_y1 = bbox_y1 - enlarge
        bbox_y2 = bbox_y2 + enlarge
        bbox_h = bbox_w
    else:
        bbox_x1 = bbox_x1 - enlarge
        bbox_x2 = bbox_x2 + enlarge
        bbox_w = bbox_h
    
    assert bbox_w == bbox_h
    img = img.crop((bbox_x1 - 0.2*bbox_w, bbox_y1 - 0.2*bbox_h, bbox_x2 + 0.2*bbox_w, bbox_y2 + 0.2*bbox_h))
    
    img = self.transformer(img) 
    label =  self.train_data[index]['category_id']
    return img, label


def train(args):
  train_loader = DataLoader(LesionTrainDataset(args.train_image_path, args.train_notes_file), batch_size = args.batch_size, shuffle=True, num_workers=24)
  val_loader = DataLoader(LesionValDataset(args.val_image_path, args.val_notes_file), batch_size = args.batch_size, shuffle=False, num_workers=8)
  # Define Model
  m = models.resnet50(pretrained = True)
  m.fc = nn.Linear(in_features=2048, out_features=args.cls_num, bias=True)
  nn.init.kaiming_normal_(m.fc.weight)
  nn.init.constant_(m.fc.bias, 0.)

  # Initialize Model
  m.cuda()
  max_epoch = args.max_epoch
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(m.parameters(), lr = args.learning_rate, momentum     = args.momentum, weight_decay = m.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch*len(train_loader))

  val_max_acc = -1

  for epoch in range(max_epoch):
    m.train()
    for i, (data, label) in enumerate(train_loader):
      data = data.cuda(non_blocking=True)
      label = label.cuda(non_blocking=True).long()
      logits = m(data)
      loss = criterion(logits, label)

      if i % 2 == 0:
        acc = (logits.topk(1, dim=1)[1] == label.view([-1,1])).sum(dim=1).float().mean()
        print('Epoch [%d][%d / %d]\tLoss:%f\tAcc@1:%.2f %%\t LR: %f' %
            (epoch, i, len(train_loader), loss.item(), acc.item()*100, optimizer.param_groups[0]['lr']))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler.step()

    m.eval()
    hit, tot = 0, 0
    score_list = []
    ground_truth_list = []
    for i, (data, label) in enumerate(val_loader):
      with torch.no_grad():
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True).long()
        logits = m(data)
        hit += (logits.topk(1, dim=1)[1] == label.view([-1,1])).sum().item()
        tot += logits.size(0)
        score_list.append(torch.softmax(logits, dim=1).cpu())
        ground_truth_list.append(label.cpu())

    ### could use for calculating the metric: roc, auc etc.
    total_ground_truth_list = torch.cat(ground_truth_list, 0)
    total_score_list = torch.cat(score_list, 0)

    if hit / tot > val_max_acc:
      val_max_acc = hit / tot
      print(' * ', end='')
      torch.save(m.state_dict(), args.save_path + '/val_best'+str(epoch)+'_params.pkl')

    print('Epoch %d Validation Accuracy: %.2f %% (max %.2f %%);' % (epoch, hit / tot * 100, val_max_acc * 100))

    if epoch % 15 == 0:
      torch.save(m.state_dict(), args.save_path + '/' +str(epoch)+'_params.pkl')

  torch.save(m.state_dict(), args.save_path + '/final_params.pkl')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Model utilizes both clinical information and image information')
  parser.add_argument("--batch_size", default = 128, type = int, help = 'Batch size for train and validation dataset')
  parset.add_argument("--epochs", default = 50, type = int, help = 'Epochs for training')
  parser.add_argument("--learning_rate", default = 0.1, type = float, help = 'Learning rate for optimization.')
  parser.add_argument("--weight_decay", default = 1e-4, type = float, help = 'L2 weight decay for optimization.')
  parser.add_argument("--momentum", default = 0.9, type = float, help = 'Momentum for optimization')
  parser.add_argument("--train_image_path", default = './Data/Images/Private/Train/', type = str, help = 'Train images folder')
  parser.add_argument("--val_image_path", default = './Data/Images/Private/Val/', type = str, help = 'Validation images folder')
  parser.add_argument("--train_image_anno", default = './Data/Annotations/Lesion/Train/', type = str, help = 'Train images annotations')
  parser.add_argument("--val_image_anno", default = './Data/Annotations/Lesion/Val/', type = str, help = 'Val images annotations')
  parser.add_argument("--cls_num", default = 2, type = int, help = 'The number of classes for classification model')
  parser.add_argument("--save_path", default = '/Checkpoint/Only_Lesion', type = str, help = 'Checkpoints save path for Lesion classification model')
  args = parser.parse_args()
  train(args)

