from tqdm import tqdm
from google.colab import drive
import numpy as np
from PIL import Image
import os

drive.mount('/content/drive')


def load_train(layer):
  assert(isinstance(layer, str))
  TRAIN = 512
  TRAIN_ADD = 384
  PATH = '/content/drive/My Drive/Rosneft Seismic/train/'
  
  x = []
  y = []
  mask = []
  print('LOAD TRAIN DATA')
  print('--load inline images')
  
  for i in tqdm(range(TRAIN)):
    
    img = np.array(Image.open(PATH+'images/inline_'+str(i+3160)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    img -= img.mean()
    img /= img.std()
    x.append(img)
    
    y.append(np.array(Image.open(PATH+'answer/'+layer+'/inline_'+str(i+3160)+'.png')))
    mask.append(np.array(Image.open(PATH+'answer/'+layer+'_border/inline_'+str(i+3160)+'.png')))

  print('--end load inline images')
  print('--load xline images')

  for i in tqdm(range(TRAIN_ADD)):
    
    img = np.array(Image.open(PATH+'images/xline_'+str(i+2017)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    
    img_left = img[:, 0:384]
    img_left -= img_left.mean()
    img_left /= img_left.std()
    x.append(img_left)

    img_right = img[:, 128:512]
    img_right -= img_right.mean()
    img_right /= img_right.std()
    x.append(img_right)

    ans = np.array(Image.open(PATH+'answer/'+layer+'/xline_'+str(i+2017)+'.png'))

    ans_left = ans[:, 0:384]
    y.append(ans_left)
    
    ans_right = ans[:, 128:512]
    y.append(ans_right)

    msk = np.array(Image.open(PATH+'answer/'+layer+'_border/xline_'+str(i+2017)+'.png'))

    msk_left = msk[:, 0:384]
    mask.append(ans_left)
    
    msk_right = msk[:, 128:512] 
    mask.append(ans_right)

  print('--end load xline images')
  print('--load turn inline images')
  
  for i in tqdm(range(TRAIN)):
    
    img = np.array(Image.open(PATH+'images/turninline_'+str(i+3160)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    img -= img.mean()
    img /= img.std()
    x.append(img)
    
    y.append(np.array(Image.open(PATH+'answer/'+layer+'/turninline_'+str(i+3160)+'.png')))
    mask.append(np.array(Image.open(PATH+'answer/'+layer+'_border/turninline_'+str(i+3160)+'.png')))

  print('--end load turn inline images')
  print('--load turn xline images')

  for i in tqdm(range(TRAIN_ADD)):
    
    img = np.array(Image.open(PATH+'images/turnxline_'+str(i+2017)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    
    img_left = img[:, 0:384]
    img_left -= img_left.mean()
    img_left /= img_left.std()
    x.append(img_left)

    img_right = img[:, 128:512]
    img_right -= img_right.mean()
    img_right /= img_right.std()
    x.append(img_right)

    ans = np.array(Image.open(PATH+'answer/'+layer+'/turnxline_'+str(i+2017)+'.png'))

    ans_left = ans[:, 0:384]
    y.append(ans_left)
    
    ans_right = ans[:, 128:512]
    y.append(ans_right)

    msk = np.array(Image.open(PATH+'answer/'+layer+'_border/turnxline_'+str(i+2017)+'.png'))

    msk_left = msk[:, 0:384]
    mask.append(ans_left)
    
    msk_right = msk[:, 128:512] 
    mask.append(ans_right)

  print('--end load turn xline images')
  print('--load noise xline images')

  for i in tqdm(range(TRAIN_ADD)):
    
    img = np.array(Image.open(PATH+'images/noisexline_'+str(i+2017)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    
    img_left = img[:, 0:384]
    img_left -= img_left.mean()
    img_left /= img_left.std()
    x.append(img_left)

    img_right = img[:, 128:512]
    img_right -= img_right.mean()
    img_right /= img_right.std()
    x.append(img_right)

    ans = np.array(Image.open(PATH+'answer/'+layer+'/xline_'+str(i+2017)+'.png'))

    ans_left = ans[:, 0:384]
    y.append(ans_left)
    
    ans_right = ans[:, 128:512]
    y.append(ans_right)

    msk = np.array(Image.open(PATH+'answer/'+layer+'_border/xline_'+str(i+2017)+'.png'))

    msk_left = msk[:, 0:384]
    mask.append(ans_left)
    
    msk_right = msk[:, 128:512] 
    mask.append(ans_right)

  print('--end load noise xline images')
  print('--load sinus inline images')
  
  for i in tqdm(range(TRAIN)):
    
    img = np.array(Image.open(PATH+'images/sinus_inline_'+str(i+3160)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    img -= img.mean()
    img /= img.std()
    x.append(img)
    
    y.append(np.array(Image.open(PATH+'answer/'+layer+'/sinus_inline_'+str(i+3160)+'.png')))
    mask.append(np.array(Image.open(PATH+'answer/'+layer+'_border/sinus_inline_'+str(i+3160)+'.png')))

  print('--end load sinus inline images')
  
  print('END LOAD TRAIN DATA') 
  print('STAR PREPROCCECING')
  
  x = np.array(x)
  x = x[:, :, :, np.newaxis]

  y = np.array(y)
  mask = np.array(mask)
  
  y = y[:, :, :, np.newaxis, np.newaxis]
  mask = mask[:, :, :, np.newaxis, np.newaxis]
  
  y = np.concatenate((y, mask), axis = 3)
  y = y.astype('float32')
  y /= 255
  print('END PREPROCCECING')
  
  return x, y

def load_start_train(layer):
  assert(isinstance(layer, str))
  TRAIN = 512
  TRAIN_ADD = 384
  PATH = '/content/drive/My Drive/Rosneft Seismic/train/'
  
  x = []
  y = []
  mask = []
  print('LOAD TRAIN DATA')
  print('--load inline images')
  
  for i in tqdm(range(TRAIN)):
    
    img = np.array(Image.open(PATH+'images/inline_'+str(i+3160)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    img -= img.mean()
    img /= img.std()
    x.append(img)
    
    y.append(np.array(Image.open(PATH+'answer/'+layer+'/inline_'+str(i+3160)+'.png')))
    mask.append(np.array(Image.open(PATH+'answer/'+layer+'_border/inline_'+str(i+3160)+'.png')))

  print('--end load inline images')
  print('--load xline images')
  for i in tqdm(range(TRAIN_ADD)):
    
    img = np.array(Image.open(PATH+'images/xline_'+str(i+2017)+'.png'))
    img = img.astype('float32')
    img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
    
    img_left = img[:, 0:384]
    img_left -= img_left.mean()
    img_left /= img_left.std()
    x.append(img_left)

    img_right = img[:, 128:512]
    img_right -= img_right.mean()
    img_right /= img_right.std()
    x.append(img_right)

    ans = np.array(Image.open(PATH+'answer/'+layer+'/xline_'+str(i+2017)+'.png'))

    ans_left = ans[:, 0:384]
    y.append(ans_left)
    
    ans_right = ans[:, 128:512]
    y.append(ans_right)

    msk = np.array(Image.open(PATH+'answer/'+layer+'_border/xline_'+str(i+2017)+'.png'))

    msk_left = msk[:, 0:384]
    mask.append(ans_left)
    
    msk_right = msk[:, 128:512] 
    mask.append(ans_right)
  print('--end load xline images')
  print('END LOAD TRAIN DATA') 
  print('STAR PREPROCCECING')
  
  x = np.array(x)
  x = x[:, :, :, np.newaxis]

  y = np.array(y)
  mask = np.array(mask)
  
  y = y[:, :, :, np.newaxis, np.newaxis]
  mask = mask[:, :, :, np.newaxis, np.newaxis]
  
  y = np.concatenate((y, mask), axis = 3)
  y = y.astype('float32')
  y /= 255
  print('END PREPROCCECING')
  
  return x, y

def load_test():
  x = []
  mass_name = []
  PATH = '/content/drive/My Drive/Rosneft Seismic/test/images/'
  dir_list = os.listdir(PATH)
  for name in tqdm(dir_list):
    if (name[-4: -1]+name[-1]) == '.png':
      img = np.array(Image.open(PATH+name))
      img = img.astype('float32')
      img = (img[:, :, 0]+img[:, :, 1]+img[:, :, 2])/3
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



