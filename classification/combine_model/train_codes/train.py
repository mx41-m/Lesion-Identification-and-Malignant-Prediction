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

class LesionTrainDataset(Dataset):
  def __init__(self, image_path, notes_file, clinical_notes_file):
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
    self.clinical_data = np.load(clinical_notes_file, allow_pickle=True).item()

  def __len__(self):
    return len(self.train_data)

  def __getitem__(self, index):
    img_name = self.train_data[index]['image_id']
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
    box = img.crop((bbox_x1 - 0.2*bbox_w, bbox_y1 - 0.2*bbox_h, bbox_x2 + 0.2*bbox_w, bbox_y2 + 0.2*bbox_h))

    box = self.transformer(box)
    label = self.train_data[index]['category_id']
    return box, label, torch.Tensor(self.clinical_data[self.train_data[index]['image_id']])

class LesionValDataset(Dataset):
  def __init__(self, image_path, notes_file, clinical_notes_file):
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
    self.clinical_data = np.load(clinical_notes_file, allow_pickle=True).item()

  def __len__(self):
    return len(self.val_data)

  def __getitem__(self, index):
    img_name = self.val_data[index]['image_id']
    img = Image.open(self.img_path + img_name)
    
    bbox_x1 = self.val_data[index]['bbox'][0] 
    bbox_y1 = self.val_data[index]['bbox'][1]
    bbox_x2 = bbox_x1 + self.val_data[index]['bbox'][2] 
    bbox_y2 = bbox_y1 + self.val_data[index]['bbox'][3]
    bbox_w = self.val_data[index]['bbox'][2]
    bbox_h = self.val_data[index]['bbox'][3]
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
    label = self.val_data[index]['category_id']
    return box, label, torch.Tensor(self.clinical_data[self.val_data[index]['image_id']])

def train(args):
  ### Dataloader
  train_dataset = LesionTrainDataset(args.train_image_path, args.train_image_anno, args.train_clinical_feat)
  val_dataset = LesionValDataset(args.val_image_path, args.val_image_anno, args.val_clinical_feat)

  train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=8)
  val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8)

  # Image feature extractor
  imgfeat_model = models.resnet50(pretrained = True)
  imgfeat_model.fc = nn.Linear(in_features=2048, out_features = args.img_cls, bias=True)
  imgfeat_model.load_state_dict(torch.load(args.img_model_ck))
  imgfeat_model.cuda()
  imgfeat_model.fc = nn.Identity()

  ### clinical linear
  clinical_combine_linear = nn.Linear(in_features = 2048 + len(val_dataset[0][-1]), out_features = args.cls_num, bias=True)
  nn.init.kaiming_normal_(clinical_combine_linear.weight)
  nn.init.constant_(clinical_combine_linear.bias, 0.)
  clinical_combine_linear.cuda()

  # other parameter
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(clinical_combine_linear.parameters(), lr = args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch*len(train_loader))

  val_max_acc = -1

  imgfeat_model.eval()
  for epoch in range(max_epoch):
    clinical_combine_linear.train()
    for i, (data, label, clinical_data) in enumerate(train_loader):
      data = data.cuda(non_blocking=True)
      label = label.cuda(non_blocking=True).long()
      clinical_data = clinical_data.cuda(non_blocking=True)
      image_logits = imgfeat_model(data)
      combine_input = torch.cat((image_logits, clinical_data),1).cuda(non_blocking=True)
      logits = clinical_combine_linear(combine_input)
      loss = criterion(logits, label)

      if i % 2 == 0:
        acc = (logits.topk(1, dim=1)[1] == label.view([-1,1])).sum(dim=1).float().mean()
        print('Epoch [%d][%d / %d]\tLoss:%f\tAcc@1:%.2f %%\t LR: %f' %
            (epoch, i, len(train_loader), loss.item(), acc.item()*100, optimizer.param_groups[0]['lr']))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler.step()

    clinical_combine_linear.eval()

    hit, tot = 0, 0
    score_list = []
    ground_truth_list = []
    for i, (data, label, clinical_data) in enumerate(val_loader):
      with torch.no_grad():
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True).long()
        clinical_data = clinical_data.cuda(non_blocking=True)
        image_logits = imgfeat_model(data)
        combine_input = torch.cat((image_logits, clinical_data),1).cuda(non_blocking=True)
        logits = clinical_combine_linear(combine_input)
        hit += (logits.topk(1, dim=1)[1] == label.view([-1,1])).sum().item()
        tot += logits.size(0)
        score_list.append(torch.softmax(logits, dim=1).cpu())
        ground_truth_list.append(label.cpu())

    if hit / tot > val_max_acc:
      val_max_acc = hit / tot
      print(' * ', end='')
      torch.save(clinical_combine_linear.state_dict(), args.save_path + '/val_best'+str(epoch)+'_params.pkl')

    print('Epoch %d Val Accuracy: %.2f %% (max %.2f %%);' % (epoch, hit / tot * 100, val_max_acc * 100))

    if epoch % 15 == 0:
      torch.save(clinical_combine_linear.state_dict(), args.save_path + '/'+str(epoch)+'_params.pkl')
  torch.save(clinical_combine_linear.state_dict(), args.save_path + '/final_params.pkl')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Model utilizes both clinical information and image information')
  parser.add_argument("--batch_size", default = 128, type = int, help = 'Batch size for training and validation')
  parset.add_argument("--epochs", default = 50, type = int, help = 'Epochs for training')
  parser.add_argument("--learning_rate", default = 0.1, type = float, help = 'Learning rate for optimization.')
  parser.add_argument("--weight_decay", default = 1e-4, type = float, help = 'L2 weight decay for optimization.')
  parser.add_argument("--momentum", default = 0.9, type = float, help = 'Momentum for optimization')  
  parser.add_argument("--train_image_path", default = './Data/Images/Private/Train/', type = str, help = 'Train images folder')
  parser.add_argument("--val_image_path", default = './Data/Images/Private/Val/', type = str, help = 'Val images folder') 
  parser.add_argument("--train_image_anno", default = './Data/Annotations/Train/', type = str, help = 'Train images annotations')
  parser.add_argument("--val_image_anno", default = './Data/Annotations/Val/', type = str, help = 'Val images annotations')
  parser.add_argument("--train_clinical_feat", default = './Data/Clinical_Features/Train/', type = str, help = 'Train clinical features folder')
  parser.add_argument("--val_clinical_feat", default = './Data/Clinical_Features/Val/', typr = str, help = 'Val clinical features folder')
  parser.add_argument("--img_cls", default = 2, type = int, help = 'The number of classes for image feature extractor')
  parser.add_argument("--cls_num", default = 2, type = int, help = 'The number of classes for combine model')
  parser.add_argument("--img_model_ck", default = '/Checkpoint/Image_model/', type = str, help = 'Checkpoints path for image features extractor model')
  parser.add_argument("--save_path", default = '/Checkpoint/Combine_Model/', type = str, help = 'Checkpoints save path for combine model')
  args = parser.parse_args()
  train(args)
