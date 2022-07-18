"""
Execute this function to segment cells in the image
---------------------------------------------------
Usage
=====
--> This function expects image in tiff format.
--> Put the tif image in "Data/Run/img" folder, while label in "Data/Run/cells". The names of the image and labels
    should be the same.
--> For comparison put the test label in "Data/Run/cells" folder.
--> The resulted images are saved in "Data/Run/output" folder.

Example :
segment(file_name = '4.tif', compare= True)
"""

import os
import skimage.io as io
import tensorflow.keras as k
from loss import *
from metrics import *
import cv2
import matplotlib.pyplot as plt


def segment(file_name, compare= True):
    # Setting paths.
    Main_Path = os.path.dirname(os.path.abspath(__file__))
    Data_Folder = Main_Path + '/Data/Run/'
    Model = Main_Path + '/Models/Pre-Trained_DANNucNet'

    # Load Model
    model = k.models.load_model(Model + '.h5', custom_objects={'loss': loss, 'f1_score1': f1_score})
    model.load_weights(Model + '.hdf5')

    img = io.imread(Data_Folder + '/img/' + file_name, as_gray=False)

    result = model.predict(np.expand_dims(img / 255., axis=0), steps=None, callbacks=None, max_queue_size=1)
    result = np.squeeze(np.squeeze(result, axis=0), axis=2)
    result[result < 0.5] = 0
    result[result >= 0.5] = 255
    cv2.imwrite(Data_Folder + 'output/cells_' + file_name, result)

    tmp_img = np.zeros((256, 256, 3), dtype=int)
    tmp_img[:, :, 2] = result
    Overlay = np.ubyte(0.6 * img + 0.4 * tmp_img)
    cv2.imwrite(Data_Folder + 'output/overlay_' + file_name, Overlay)

    if compare is True:
        label = io.imread(Data_Folder + '/cells/' + file_name, as_gray=True)
        tmp_img[:, :, 1] = label
        cv2.imwrite(Data_Folder + 'output/diff_' + file_name, tmp_img)

        titles = ['Original Image', 'Cells Segmented', 'False Neg(Green), False Pos(Red), True Pos(Yellow)']
        ax = plt.subplot(1, 3, 1)
        plt.axis('off')

        ax.set_title(titles[0])
        plt.imshow(img)
        ax = plt.subplot(1, 3, 2)
        plt.axis('off')
        ax.set_title(titles[1])

        plt.imshow(Overlay)

        ax = plt.subplot(1, 3, 3)
        ax.set_title(titles[2])
        plt.axis('off')
        b, g, r = cv2.split(tmp_img)       # get b,g,r
        rgb_img = cv2.merge([r, g, b])     # switch it to rgb
        plt.imshow(rgb_img)

        plt.show()
        plt.close()
        plt.clf()
    else:
        titles = ['Original Image', 'Cells Segmented']
        ax = plt.subplot(1, 2, 1)
        plt.axis('off')

        ax.set_title(titles[0])
        plt.imshow(img)

        ax = plt.subplot(1, 2, 2)
        ax.set_title(titles[1])
        plt.axis('off')
        plt.imshow(Overlay)

        plt.show()
        plt.close()
        plt.clf()


segment(file_name = 'BL_1595.tif', compare = True)

