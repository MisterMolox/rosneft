from tqdm import tqdm, tqdm_notebook
from google.colab import drive
import numpy as np
from PIL import Image
import os, itertools
from albumentations import OneOf, Cutout

drive.mount('/content/drive')


def load_data(layer, val = False):
  assert(isinstance(layer, str))
  TRAIN = 512
  TRAIN_ADD = 384
  VALID_IN = 50
  VALID_X = 39
  PATH = '/content/drive/My Drive/Rosneft Seismic/train/'
  augm = OneOf([
    Cutout(num_holes=4, max_h_size=38, max_w_size=38, always_apply=True),
    Cutout(num_holes=50, max_h_size=6, max_w_size=6, always_apply=True),
    Cutout(num_holes=20, max_h_size=15, max_w_size=15, always_apply=True)
  ], p=1.)

  x = []
  y = []
  borders = []
  if val:
    inds_in = list(itertools.chain(range(0, VALID_IN), range(TRAIN-VALID_IN, TRAIN)))
    inds_x = list(itertools.chain(range(0, VALID_X), range(TRAIN_ADD-VALID_X, TRAIN_ADD)))
    print('LOAD VALIDATION DATA')
  else:
    inds_in = list(range(VALID_IN, TRAIN-VALID_IN))
    inds_x = list(range(VALID_X, TRAIN_ADD-VALID_X))
    print('LOAD TRAIN DATA')
  
  print('--load inline images')
  
  for i in tqdm_notebook(inds_in):
    
    img = np.array(Image.open(PATH+'images/inline_'+str(i+3160)+'.png'))
    mask = np.array(Image.open(PATH+'answer/'+layer+'/inline_'+str(i+3160)+'.png'))
    border = np.array(Image.open(PATH+'answer/'+layer+'_border/inline_'+str(i+3160)+'.png'))

    img = img.astype('float32')
    img = img[:, :, 0]*0.299 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114
    img -= img.mean()
    img /= img.std()

    x.append(img)
    y.append(mask)
    borders.append(border)
    if not val:
      x.append(img[:, ::-1])
      y.append(mask[:, ::-1])
      borders.append(border[:, ::-1])
      
      x.append(augm(image=img)["image"])
      y.append(mask)
      borders.append(border)
  
  print('--load xline images')

  for i in tqdm_notebook(inds_x):
    
    img = np.array(Image.open(PATH+'images/xline_'+str(i+2017)+'.png'))
    mask = np.array(Image.open(PATH+'answer/'+layer+'/xline_'+str(i+2017)+'.png'))
    border = np.array(Image.open(PATH+'answer/'+layer+'_border/xline_'+str(i+2017)+'.png'))

    img = img.astype('float32')
    img = img[:, :, 0]*0.299 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114
    img_left = img[:, 0:384]
    img_left -= img_left.mean()
    img_left /= img_left.std()
    img_right = img[:, 128:512]
    img_right -= img_right.mean()
    img_right /= img_right.std()
    mask_left = mask[:, 0:384]
    mask_right = mask[:, 128:512]
    border_left = border[:, 0:384]
    border_right = border[:, 128:512]

    x.append(img_left)
    x.append(img_right)
    y.append(mask_left)
    y.append(mask_right)
    borders.append(border_left)
    borders.append(border_right)
    if not val:
      x.append(img_left[:, ::-1])
      x.append(img_right[:, ::-1])
      y.append(mask_left[:, ::-1])
      y.append(mask_right[:, ::-1])
      borders.append(border_left[:, ::-1])
      borders.append(border_right[:, ::-1])
      x.append(augm(image=img_left)["image"])
      x.append(augm(image=img_right)["image"])
      y.append(mask_left)
      y.append(mask_right)
      borders.append(border_left)
      borders.append(border_right)

  if val:
    pass
    print('END LOAD VALIDATION DATA')
  else:
    print('END LOAD TRAIN DATA')

  print('STAR PREPROCCECING')
  
  x = np.array(x)
  x = x[:, :, :, np.newaxis]

  y = np.array(y)
  borders = np.array(borders)
  
  y = y[:, :, :, np.newaxis, np.newaxis]
  borders = borders[:, :, :, np.newaxis, np.newaxis]
  
  y = np.concatenate((y, borders), axis = 3)
  y = y.astype('float32')
  y /= 255
  print('END PREPROCCECING')
  
  return x, y


def load_test():
  x = []
  mass_name = []
  PATH = '/content/drive/My Drive/Rosneft Seismic/test/images/'
  dir_list = os.listdir(PATH)
  for name in tqdm_notebook(dir_list):
    if (name[-4: -1]+name[-1]) == '.png':
      img = np.array(Image.open(PATH+name))
      img = img.astype('float32')
      img = img[:, :, 0]*0.299 + img[:, :, 1]*0.587 + img[:, :, 2]*0.114
      x.append(img)
      mass_name.append(name)
  return mass_name, x 


def cut_test_img(x):
  
  #Input images (x[384, m]) have a different format m
  #Divide images into k (or k+1) small images with shapes (384, 384)
  x = np.array(x)
  x = x.astype('float32')
  m = x.shape[1]
  x = np.array(x)
  y = []
  k = m // 384

  for i in range(k):
    t = x[:, i*384 : 384*(i+1)]
    t -= t.mean()
    t /= t.std()
    y.append(t)

  if (m%384) > 0:
    t = x[:, m-384 : m]
    t -= t.mean()
    t /= t.std()
    y.append(t)
  
  y = np.array(y)
  y = y[:, :, :, np.newaxis]
  return m, y

def connect_test_img(m, x):

  #Concatenate the resulting masks into one image(y) with shape (384, m)
  
  k = m // 384
  if (m == 384): y = x
  else:
    y = np.copy(x[0])
    for i in range(1, k):
      y = np.concatenate((y, x[i]), axis = 1)
    if (m%384 > 0):
      y = y[:, 0 : m-384]
      y = np.concatenate((y, x[k]), axis = 1)
  return y



