"""
Execute test.py to test the model.
A trained model can be found in Model folder, named as, "Pre-Trained_DANNucNet". Use predict.py to use our trained
model.
----------------------------------------------------------------------------------------------------------------------
Usage
=====
--> test.py expects image in tiff format.
--> Put processed test data in the "Data/PanNuke_Test_Data" folder. The data is organised as,
    -- cells
        -- Adrena_Gland
        -- Bladder
        -- .
        -- .
        -- ...
    -- image
        -- Adrena_Gland
        -- Bladder
        -- .
        -- .
        -- ...
--> Both numerical and pictorial results are stored in the "Result" folder

"""

import os
import skimage.io as io
import tensorflow.keras as k
from loss import *
from metrics import *
import cv2

# Setting paths.
Main_Path = os.path.dirname(os.path.abspath(__file__))
Data_Folder = Main_Path + '/Data/PanNuke_Test_Data/'
Result_Folder = Main_Path + '/Results/'
Model_Folder = Main_Path + '/Models/'
NetName = 'Pre-Trained_DANNucNet'

# PanNuke Data : The tissues are stored in separate folders after pre-processing
Tissue_List = ["Adrenal_Gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon", "Esophagus", "HeadNeck", "Kidney",
               "Liver", "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus"]

# Load Model
model = k.models.load_model(Model_Folder + NetName + '.h5', custom_objects={'loss': loss, 'f1_score1': f1_score})
model.load_weights(Model_Folder + NetName + '.hdf5')

All_Tissue_results = [[0] * 4] * (len(Tissue_List) + 2)
FAvg_F1 = [[0] * 1] * (len(Tissue_List))
FAvg_JDI = [[0] * 1] * (len(Tissue_List))
FAvg_Precision = [[0] * 1] * (len(Tissue_List))
FAvg_Recall = [[0] * 1] * (len(Tissue_List))
tmp_img = np.zeros((256, 256, 3), dtype=int)
olay_img = np.zeros((256, 256, 3), dtype=int)


for Tiss in range(0, len(Tissue_List)):
    Tissue = Tissue_List[Tiss]
    print("Processing Tissue --> ", Tissue_List[Tiss])
    Img_Path = Data_Folder + '/img/' + Tissue
    Label_Path = Data_Folder + '/cells/' + Tissue

    No_of_images = len(os.listdir(Img_Path))
    print("Total Test Images found --> ", No_of_images)

    ind_F1 = np.zeros((No_of_images,), dtype=float)
    ind_JI = np.zeros((No_of_images,), dtype=float)
    ind_Precision = np.zeros((No_of_images,), dtype=float)
    ind_Recall = np.zeros((No_of_images,), dtype=float)
    lp = 0

    Img_files  = os.listdir(os.path.join(Img_Path))
    Label_files= os.listdir(os.path.join(Label_Path))

    for i in Img_files:
        for j in Label_files:
            try:
                if i == j:
                    img = io.imread(Img_Path + "/" + i, as_gray=False) / 255.
                    label = io.imread(Label_Path + "/" + j, as_gray=True) / 255.

                    result = model.predict(np.expand_dims(img, axis=0), batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=1)
                    result = np.squeeze(np.squeeze(result, axis=0), axis=2)
                    cv2.imwrite(Result_Folder + '/images/prediction/' + Tissue + '/' + i, result * 255)
                    result[result < 0.5] = 0
                    result[result >= 0.5] = 1

                    tmp_img[:, :, 2] = result * 255
                    olay_img[:, :, 2] = result * 255
                    olay_img = np.ubyte(255*0.7 * img + 0.30 * olay_img)
                    cv2.imwrite(Result_Folder + '/images/overlay/' + Tissue + '/' + i, olay_img)

                    tmp_img[:, :, 1] = label * 255
                    cv2.imwrite(Result_Folder + '/images/diff/' + Tissue + '/' + i, np.round(tmp_img))


                    ind_JI[lp] = jaccard(label, result)
                    ind_F1[lp] = 100 * (f1_score(result, label))
                    ind_Precision[lp] = 100 * Precision(result, label)
                    ind_Recall[lp] = 100 * Recall(result, label)

                    lp = lp + 1
                    if np.isnan(ind_F1[lp - 1]):
                        lp = lp - 1
                    if ind_F1[lp - 1] == 0:
                        lp = lp - 1
            except:
                print("File not found")

    total_F1 = np.round_(np.sum(ind_F1) / lp, decimals=4, out=None)
    total_JDI = np.round_(np.sum(ind_JI) / lp, decimals=4, out=None)
    total_Precision = np.round_(np.sum(ind_Precision) / lp, decimals=4, out=None)
    total_Recall = np.round_(np.sum(ind_Recall) / lp, decimals=4, out=None)

    print("F1 , JI, Precision, Recall : ", total_F1, total_JDI, total_Precision, total_Recall)

    All_Tissue_results[Tiss+1] = [Tissue, total_F1, total_JDI, total_Precision, total_Recall]

    FAvg_F1[Tiss] = total_F1
    FAvg_JDI[Tiss] = total_JDI
    FAvg_Precision[Tiss] = total_Precision
    FAvg_Recall[Tiss] = total_Recall

All_Tissue_results[0] = ['Tissue', 'F1', 'AJI', 'Precision', 'Recall']

All_Tissue_results[len(Tissue_List)+1] = ['Average', np.sum(FAvg_F1) / len(Tissue_List),
                                          np.sum(FAvg_JDI) / len(Tissue_List),
                                          np.sum(FAvg_Precision) / len(Tissue_List),
                                          np.sum(FAvg_Recall) / len(Tissue_List)]

Test_csv_file_All_Result = Result_Folder + 'Numerical_Results/' + 'PanNuke_Tissue_Results.csv'
np.savetxt(Test_csv_file_All_Result, All_Tissue_results, delimiter=",", fmt='%s')

