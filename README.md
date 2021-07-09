# Tuberculosis-Classification
Deep Learning bootstrapping approach on National Institutes of Health TB X-ray images Classification

# Data Resource
The 2 datasets are downloaded from National Institutes of Health(NIH) IRL:https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html

# Instructions
1. Code ["File_preprocessing.py.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/File_preprocessing.py) is used for data splitting(train/test).
2. Code ["Image_preprocessing.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/Image_preprocessing.py) is used for customed image function(Crop middle/ histogram equalization).
3. Code ["train.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/train.py) is for model training using Bootstrapping method.
4. Code ["Evaluation.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/Evaluation.py) concat and evaluate all the bootstrapping result.

# Bootstrapping Approach
The bootstrapping approach is a method using the model parameters which trained by last bootstrap of model as the initial parameters of a new bootstrap.
(EX: The initial parameters of first bootstrap training model is the parameters of ResNet50, the initial parameters of the next bootstrap training model is based on the previous bootstrap of model.
