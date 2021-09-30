# NIH Tuberculosis-Classification
Deep Learning bootstrapping approach on National Institutes of Health TB X-ray images Classification

# Data Resource & Citation
The 2 datasets are downloaded from National Institutes of Health(NIH).

Jaeger, S., Candemir, S., Antani, S., Wáng, Y. X., Lu, P. X., & Thoma, G. (2014). Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. Quantitative imaging in medicine and surgery, 4(6), 475–477. https://doi.org/10.3978/j.issn.2223-4292.2014.11.20

URL:https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html

# Enviornment & Deep Learning Framework
Python, Tensorflow 2.4.0

# Instructions
1. Code ["1.File_preprocessing.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/1.File_preprocessing.py) is used for data splitting(train/test).
2. Code ["2.Image_preprocessing.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/2.Image_preprocessing.py) is used for customed image function(Crop middle/ Histogram equalization).
3. Code ["3.Train.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/3.Train.py) is for model training using Bootstrapping approach.
4. Code ["4.Evaluation.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/4.Evaluation.py) is used for concating and evaluating all the bootstrapping results.

# Bootstrapping Approach
Interpretation
---
The bootstrapping approach is a method using the model parameters which trained by the previous bootstrap of model as the initial parameters of a new bootstrap.
(EX: The initial parameters of first bootstrap training model is the parameters of ResNet50, the initial parameters of the second bootstrap training model is based on the first bootstrap of model.)

Usage
---
The bootstrapping approach can be used as a situation lacking of data(images). By sampling images in training set into sub-training set and sub-validation set over and over to save training samples. If you do not want to use bootstrapping approach, input bootstrap = 1 at the last line of ["3.Train.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/3.Train.py), you may train the model only by transfer learning(ResNet50 is setted in the source code).

# Evaluation
The result is evaluated by common screening index like Sensitivity, Specificity ..etc, which Loading Rate index is calculated by predict positive/all the predict samples under the circumstance of no predict False Negative samples(c=0 in confusion matrix).

# Graphical User Interface (GUI)
We also build a GUI to interface the model in order to make this computer assisted diagnosis system into practice. This system allows inputting DICOM(dcm) and PNG X-ray image to predict the result. 

Source Code: ["GUI.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/GUI.py) used TKinter package(Python).

The example GUI image is below.
<p align="center">
  <img src="Example image/MTBNET.png">
</p>
