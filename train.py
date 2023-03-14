"""
Execute train.py to train the model. Note that the pre-processing is done in Matlab. That part of the code will be
provided later on.
A trained model can be found in Model folder, named as, "Pre-Trained_DAN_NucNet". Use predict.py to use our
trained model.
----------------------------------------------------------------------------------------------------------------------
Usage

--> train.py expects image in tiff format.

--> Put processed training data in the "Data/" folder. The data is organised as,
    -- train
        -- cells

        -- image

    -- validation
        -- cells

        -- image

--> The resulted model and logs are stored in the respective folders

"""

import tensorflow.keras as k
from Data_adjust import *
from model import dan_nuc_net
from get_my_gpu_settings import *
set_my_gpu()

def train_net(Data_Folder = os.path.dirname(os.path.abspath(__file__)) + '/Data/PanNuke_Train_Data/',
              Model_Folder =  os.path.dirname(os.path.abspath(__file__)) + '/Models/',
              Log_Folder =  os.path.dirname(os.path.abspath(__file__)) + '/logs/',
              Net_Name = 'DAN_NucNet',
              batch_size = 5,
              epochs = 100,
              early_stop_patience = 9):

    with tf.device('/GPU:0'):
        lr_reducer = ReduceLROnPlateau(factor=0.5, patience=4, cooldown=0, min_lr=0.1e-5, verbose=1)

        data_gen_args = dict(rotation_range=0, width_shift_range=0, height_shift_range=0, shear_range=0, zoom_range=0,
                             horizontal_flip=True, vertical_flip=True, fill_mode='nearest')

        Train_data_path = Data_Folder + '/train'
        Valid_data_path = Data_Folder + '/validation'

        tot_train_imgs = len(os.listdir(Train_data_path + '/img'))
        tot_valid_imgs = len(os.listdir(Valid_data_path + '/img'))

        train_gen = data_generator(batch_size, Train_data_path, 'img', 'cells', data_gen_args, target_size=(256, 256),
                                   save_to_dir=None, image_color_mode="rgb")
        valid_gen = data_generator(batch_size, Valid_data_path, 'img', 'cells', data_gen_args, target_size=(256, 256),
                                   save_to_dir=None, image_color_mode="rgb")

        model = dan_nuc_net()

        model.save(Model_Folder + Net_Name + '_model.h5')

        EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=1, mode='min')

        model_checkpoint = ModelCheckpoint(Model_Folder + Net_Name + '_membrane.hdf5', monitor='val_loss',
                                           patience=early_stop_patience, verbose=2, save_best_only=True)

        callbacks = [lr_reducer, model_checkpoint,
                     k.callbacks.TensorBoard(log_dir=Log_Folder, histogram_freq=0, batch_size=batch_size, write_graph=False,
                                             write_grads=False,  write_images=False, embeddings_freq=0, update_freq='epoch',
                                             embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None),
                     EarlyStop, ClearMemory()]

        model.fit(train_gen, steps_per_epoch=tot_train_imgs // batch_size, epochs=epochs,  validation_data=valid_gen,
                  max_queue_size=2, validation_steps=tot_valid_imgs // batch_size, callbacks=callbacks, shuffle=True,
                  workers=1, verbose=1, validation_freq=1)
