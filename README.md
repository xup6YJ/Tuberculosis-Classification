# Tuberculosis-Classification
Deep Learning bootstrapping approach on National Institutes of Health TB X-ray images Classification

# Data Resource
The 2 datasets are downloaded from National Institutes of Health(NIH).

URL:https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html

# Instructions
1. Code ["File_preprocessing.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/File_preprocessing.py) is used for data splitting(train/test).
2. Code ["Image_preprocessing.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/Image_preprocessing.py) is used for customed image function(Crop middle/ Histogram equalization).
3. Code ["train.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/train.py) is for model training using Bootstrapping method.
4. Code ["Evaluation.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/Evaluation.py) is used for concating and evaluating all the bootstrapping results.

# Bootstrapping Approach
Interpretation
---
The bootstrapping approach is a method using the model parameters which trained by the previous bootstrap of model as the initial parameters of a new bootstrap.
(EX: The initial parameters of first bootstrap training model is the parameters of ResNet50, the initial parameters of the second bootstrap training model is based on the first bootstrap of model.)

Usage
---
The bootstrapping approach can be used as a situation lacking of data(images). By sampling images in training set into sub-training set and sub-validation set over and over to save training samples.

# Evaluation
The result is evaluated by common screening index like Sensitivity, Specificity ..etc, which Loading Rate index is calculated by predict positive/all the predict samples under the circumstance of no predict False Negative samples(c=0 in confusion matrix).
