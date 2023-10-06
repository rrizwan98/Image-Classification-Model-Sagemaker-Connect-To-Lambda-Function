from keras.layers import  GlobalAveragePooling2D
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras.models import Sequential, save_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout 
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import argparse, os
import pandas as pd
import numpy as np
import keras


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    batch_size = args.batch_size
    model_dir  = args.model_dir
    training_dir   = args.training
    chk_dir = '/opt/ml/checkpoints'
    # validation_dir = args.validation
    
    dataset = training_dir
    # val = validation_dir
    
#     print("Train Data Shape",train.shape)
#     print("Test Data Shape",val.shape)
    
#    Data Augmentation
# Using ImageDataGenerator to load the Images for Training and Testing the CNN Model
    datagenerator = {
        "train": ImageDataGenerator(horizontal_flip=True,
                                    vertical_flip=True,
                                    rescale=1. / 255,
                                    validation_split=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    rotation_range=30,
                                ).flow_from_directory(directory=dataset,
                                                        target_size=(300, 300),
                                                        subset='training',
                                                        ),

        "valid": ImageDataGenerator(rescale=1 / 255,
                                    validation_split=0.1,
                                ).flow_from_directory(directory=dataset,
                                                        target_size=(300, 300),
                                                        subset='validation',
                                                        ),
    }
    
    # print(train_generator.class_indices)

    
#     used ResNet50V2 model
    
    input_shape = (300,300,3)
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=input_shape, include_top=False)

    for layer in base_model.layers:
        layer.trainable = False
    # base_model.summary()
    
    # Adding some more layers at the end of the Model as per our requirement
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.15),
        Dense(1024, activation='relu'),
        Dense(5, activation='softmax') # 5 Output Neurons for 5 Classes
    ])

    # Using the Adam Optimizer to set the learning rate of our final model
    opt = optimizers.Adam(learning_rate=0.0001)

    # Compiling and setting the parameters we want our model to use
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    # Define callback to save best epoch
    chk_name = "model_{epoch:02d}-{val_accuracy:.2f}"
    checkpointer = ModelCheckpoint(filepath=os.path.join(chk_dir,chk_name),
                               monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    callbacks_list = [checkpointer]
    # Viewing the summary of the model
    model.summary()

    # Setting variables for the model

    # Seperating Training and Testing Data
    train_generator = datagenerator["train"]
    valid_generator = datagenerator["valid"]

    # Calculating variables for the model
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = valid_generator.n // batch_size

    print("steps_per_epoch :", steps_per_epoch)
    print("validation_steps :", validation_steps)
    
    #start training
    history = model.fit_generator(generator=train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=valid_generator, validation_steps=validation_steps,callbacks=callbacks_list)
    
    # save model
    # save model for Tensorflow Serving
    save_model(model, os.path.join(model_dir, '1'), save_format='tf')
    print("Model successfully saved at: {}".format(model_dir))