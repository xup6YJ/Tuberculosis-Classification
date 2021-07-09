# NIH Tuberculosis-Classification
Deep Learning bootstrapping approach on National Institutes of Health TB X-ray images Classification

# Data Resource
The 2 datasets are downloaded from National Institutes of Health(NIH).

URL:https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html

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
The bootstrapping approach can be used as a situation lacking of data(images). By sampling images in training set into sub-training set and sub-validation set over and over to save training samples. If you do not want to use bootstrapping approach, input bootstrap = 1 at the last line of ["3.Train.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/3.Train.py), you may train the model only by transfer learning.

# Evaluation
The result is evaluated by common screening index like Sensitivity, Specificity ..etc, which Loading Rate index is calculated by predict positive/all the predict samples under the circumstance of no predict False Negative samples(c=0 in confusion matrix).
