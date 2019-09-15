
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from lib import *
import random
import os

from glob import glob

def show_train_history(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='best')
  plt.show()

def analyze_dataset(train_path, val_path, test_path):

  train_files = [f for f in glob(train_path + "**/*.jpg", recursive=True)]
  train_masks = [f for f in glob(train_path +'_mask' + "**/*.png", recursive=True)]
  valid_files = [f for f in glob(val_path + "**/*.jpg", recursive=True)]
  valid_masks = [f for f in glob(val_path + '_mask' + "**/*.png", recursive=True)]
  test_files = [f for f in glob(test_path + "**/*.jpg", recursive=True)]
  print("Size of train datasest: " + str(len(train_files)) + " images and " + str(len(train_masks)) + " masks")
  print("Size of validation datasest: " + str(len(valid_files)) + " images and " + str(len(valid_masks)) + " masks")
  print("Size of test datasest: " + str(len(test_files)) + " images")
  ind = 1
  img = np.array(Image.open(f"{train_path}/{ind}.jpg"))
  mask = np.array(Image.open(f"{train_path}_mask/{ind}.png"))
  print("Image size is: " + str(img.shape[0]) + "x" + str(img.shape[1]) + "x" + str(img.shape[2]))
  print("Mask size is:  " + str(mask.shape[0]) + "x" + str(mask.shape[1]) + "x1")
  print("Each pixel in image belongs to [" + str(np.min(img)) + ", " + str(np.max(img)) + "]")
  print("Each pixel in mask belongs to  [" + str(np.min(mask)) + ", " + str(np.max(mask)) + "]")
  return train_files, train_masks, valid_files, valid_masks, test_files

def show_results(pred, x_val, y_val):
  im_id = random.randint(0,15)
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
  axes[0].imshow(x_val[im_id])

  #Делаем бинарную маску только по значениям, превысившим порог в 0.5
  axes[1].imshow(pred[im_id, ..., 0] > 0.5)
  axes[2].imshow(y_val[im_id, ..., 0])
  plt.show()

def triple_ensemble(pred1, pred2, pred3):
  ensemble_pred = []
  for i in range(pred1.shape[0]):
  
    ensemble_pred.append((pred1[i] + pred2[i] + pred3[i]) / 3.)
  return np.array(ensemble_pred)
  
def predict_Valid(model, df):
    x = []
    masks_en = []
    new_list = []
    
    name_list = os.listdir('/content/data/valid')
    path_list = list(df['img'])
    
    for i in range(df.shape[0]):
        img = np.array(Image.open(path_list[i]))
        img = cv2.resize(img, (256, 256))
        x += [img]
        
    x = np.array(x) / 255.
    mask = model.predict(x)
    resized_mask = [cv2.resize(mask[i], (240, 320)) for i in range(df.shape[0])]
    resized_mask = [(resized_mask[i] > 0.5).astype(int) for i in range(df.shape[0])]
    mask_en = [encode_rle(m) for m in resized_mask]
    
    for e in name_list:
      new_list.append(e.split('.')[0])
      
    dic = {'id': new_list, 'mask_rle': mask_en}
    return pd.DataFrame.from_dict(dic)

def predict_Test(model):
    x = []
    masks_en = []
    
    pat = f"/content/data/test"
    name_list = os.listdir(pat)
    listd = os.listdir(pat)
    
    im_path = sorted([(f"{pat}/{x}") for x in listd])
    for i in range(len(im_path)):
        img_path=im_path[i]
        img = np.array(Image.open(img_path))
        img = cv2.resize(img, (256, 256))
        x += [img]
        
    x = np.array(x) / 255.
    mask = model.predict(x)
    resized_mask = [cv2.resize(mask[i], (240, 320)) for i in range(len(im_path))]
    resized_mask = [(resized_mask[i] > 0.5).astype('uint8') for i in range(len(im_path))]
    return resized_mask