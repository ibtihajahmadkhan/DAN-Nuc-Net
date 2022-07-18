# DAN-NucNet
This project is the implementation of the paper, "DAN-NucNet : A Spatial-Channel Attention Based UNet for Nuclei Segmentation in Cancer Histology Images".
There are three main files,
* predict.py : it uses the trained model to segment the images. The example images along with the segmented images and the numerical results can be seen in the "Results" folder. We provide the trained model (link can be found in the "Model" folder, named as, "Pre-Trained_SCANucNet"), so that the viewers can reproduce the results. 
* train.py : it is used to train the proposed network. currently, the pre-processing is done using Matlab. We are working to implement it in python. The code will be shared later on.
* test.py : we used test.py to test the proposed network for various types of tissues.

___The respository is currently being updated. Full implementation will be publically available upon completion of update and paper acceptance.___
