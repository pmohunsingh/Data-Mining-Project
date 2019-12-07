# Data Mining Project
U.S. Census Income Classification

## Running the Files
The files should be run on jupyter notebook. 
The DropNa folder contains preproccessing code that converts all categorical variables into dummy variables, including the label variable, for both the test and training datasets. Run both the files for the training and test dataset. 

The Mode folder has files that perform mode imputation on both the training and test datasets and creates new csv files with the imputed data. 

The Ensemble folder contains the code that performs the classification algorithms. 
Pearson_Mode_Ensemble is the Ensemble with the Pearson Coefficient. RK_KN_LR is the mode imputed ensemble. Training Ensemble is the Ensemble where the training file is used to test the algorithms for both test and train. 

