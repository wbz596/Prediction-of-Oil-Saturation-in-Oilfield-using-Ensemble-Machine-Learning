# Prediction-of-Oil-Saturation-in-Oilfield-using-Ensemble-Machine-Learning

This project aim to develop a data-driven reservoir characterization workflow. The ensemble machine learning workflow is presented to predict the time-lapse wellbore oil saturation profile, using field-wide production and injection data.

# Data Description
1. This study utilizes data from the Volve oilfield, located in the central part of the North Sea, 200 km west of Stavanger at the southern end of the Norwegian sector.

<img src="https://raw.githubusercontent.com/wbz596/Prediction-of-Oil-Saturation-in-Oilfield-using-Ensemble-Machine-Learning/master/img/3.png" width="50%" height="50%">
2. A black-oil heterogeneous reservoir model was built with the seven producers and three injectors.The areal field map with the well locations is shown as below:

<img src="https://raw.githubusercontent.com/wbz596/Prediction-of-Oil-Saturation-in-Oilfield-using-Ensemble-Machine-Learning/master/img/2.png" width="50%" height="50%">

# Methodology
1. The objective of the ensemble machine learning model was to predict the time-lapsed oil saturation profile using the injection (water) and production (oil, water, gas) history across the field. The workflow was demonstrated for predicting saturation profiles for four producer wells: PF-15C, PF-14, PF-12, and PF-11B. 
2. The model inputs included production (oil, water and gas) and injection (water) rates from the target well and the surrounding wells, as well as the time and depth measured across the producing interval. The output was oil saturation profile at the wellbore. 
3. In order to make the time scale consistent, averaging the production and injection data points over a period of time to match corresponding time steps of oil saturation. Then removing missing data in oil saturation profiles are required for machine learning model.

# Implementation
1. Correlation matrix and Feature Importance
2. Use Scikit-learn package in Python as the framework to build the Random Forest model
3. Several parameters (n_estimators, max_features, max_samples) in the RF model were tuned to improve the predictive power and generalizability/robustness of the model. 

# Results
1. We introduce a novel simple and efficient workflow for predicting oil saturation from production and injection data which are easily and reliably to obtain. It can predict the time evolution of the saturation with simple input parameters, without resorting to reservoir simulation or elaborate geological studies. 
2. According to the 3.5 years training dataset and 1.5 years testing dataset, the results show the model predicts time-lapse oil saturation profiles at four deviated wells with over 90% accuracy and less than 0.05 RMSE. 
