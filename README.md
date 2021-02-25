# glioma-genotype-radiomics
# 1 Requirements
python >= 3.6
numpy
pandas
scipy
sklearn

R >= 3.5.0

brnn
caret
pROC

# 2 description of the above three files
1 feature_selection.py
Select features for your task, here an example for prediction of genotype in giloma was given.
Please modify Line 20-22 as the direction to your feature file (.csv)

2 redundancy_threshold.py
Class RedundancyThresholdSurv was used to remove redundancy in the feature matrix.
Place this file under the direction of "sklearn/feature_selection", my direction is "xxx/python3.6/site-packages/sklearn/feature_selection/".

3 brnn.R
construct the predictive model for your task using Bayesian Regularization Neural Networks based on the selected features.

# 3 Citation
If you find it useful for your work, please cite the following work.

