import numpy as np
import tensorflow as tf
from PIL import Image
import cv2, keras
import segmentation_models as sm
from lib import *

def keras_generator_with_augs(gen_df, batch_size, path, aug, PSPNet=False):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            img = np.array(Image.open(f"{path}/{img_name}.jpg"))  #cv2.imread(f"{path}/{img_name}.jpg")
            mask = decode_rle(mask_rle)
            if(PSPNet):
              img = cv2.resize(img, (240, 240))
              mask = cv2.resize(mask, (240, 240))
            else:
              img = cv2.resize(img, (256, 256))
              mask = cv2.resize(mask, (256, 256))
            
            augmented = aug(image=img, mask=mask)

            image_aug = augmented['image']
            mask_aug = augmented['mask']
            
            x_batch += [img]
            y_batch += [mask]
            x_batch += [image_aug]
            y_batch += [mask_aug]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)
        
def keras_generator(gen_df, batch_size, path, PSPNet=False):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            img = np.array(Image.open(f"{path}/{img_name}.jpg"))  #cv2.imread(f"{path}/{img_name}.jpg")
            mask = decode_rle(mask_rle)
            
            if(PSPNet):
              img = cv2.resize(img, (240, 240))
              mask = cv2.resize(mask, (240, 240))
            else:
              img = cv2.resize(img, (256, 256))
              mask = cv2.resize(mask, (256, 256))
            
            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)        

def init_and_train_model(train_df, train_path, val_df, val_path, model_type, BACKBONE, AUGMENTATIONS, batch_size, epoch_num):
  preprocess_input = sm.backbones.get_preprocessing(BACKBONE)
  
  if(model_type == "Linknet"):
    model = sm.Linknet(BACKBONE,input_shape = (256, 256, 3), encoder_weights='imagenet', encoder_freeze=True)
    train_gen = keras_generator_with_augs(train_df, batch_size, train_path, AUGMENTATIONS)
    val_gen = keras_generator(val_df, batch_size, val_path)
  elif(model_type == "Unet"):
    model = sm.Unet(BACKBONE,input_shape = (256, 256, 3), encoder_weights='imagenet', encoder_freeze=True)
    train_gen = keras_generator_with_augs(train_df, batch_size, train_path, AUGMENTATIONS)
    val_gen = keras_generator(val_df, batch_size, val_path)
  elif(model_type == "FPN"):
    model = sm.FPN(BACKBONE,input_shape = (256, 256, 3), encoder_weights='imagenet', encoder_freeze=True)
    train_gen = keras_generator_with_augs(train_df, batch_size, train_path, AUGMENTATIONS)
    val_gen = keras_generator(val_df, batch_size, val_path)
  elif(model_type == "PSPNet"):
    model = sm.PSPNet(BACKBONE,input_shape = (240, 240, 3), encoder_weights='imagenet', encoder_freeze=True)
    train_gen = keras_generator_with_augs(train_df, batch_size, train_path, AUGMENTATIONS, PSPNet=True)
    val_gen = keras_generator(val_df, batch_size, val_path, PSPNet = True)
    
  model.compile(
    'Adam',
    loss=sm.losses.dice_loss,
    metrics=[sm.metrics.dice_score],
  )
  
  best_w = keras.callbacks.ModelCheckpoint(model_type + '_' + BACKBONE + '_best.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

  last_w = keras.callbacks.ModelCheckpoint(model_type + '_' + BACKBONE + '_last.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=False,
                                save_weights_only=True,
                                mode='auto',
                                period=1)


  callbacks = [best_w, last_w]
  
  history = model.fit_generator(train_gen,
              steps_per_epoch=50,
              epochs=epoch_num,
              verbose=1,
              callbacks=callbacks,
              validation_data=val_gen,
              validation_steps=50,
              class_weight=None,
              max_queue_size=1,
              workers=1,
              use_multiprocessing=False,
              shuffle=True,
              initial_epoch=0)
  return model, history
  