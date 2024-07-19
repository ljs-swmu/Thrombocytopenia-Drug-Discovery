Overview
========================
Thrombocytopenia, linked to thrombopoietin (TPO) deficiency, has limited treatment options and poses significant health risks. Traditional screening methods have struggled to find new therapeutic agents for this condition. In this study, we utilize machine learning techniques to identify potential drugs that promote megakaryocyte differentiation and platelet production. By developing 112 classifiers using eight different algorithms and fourteen molecular features, we aim to uncover the structural characteristics associated with hematopoietic activity. Finally, we selected 379 optimized features to build a predictive model based on the random forest algorithm to forecast hematopoietic activity.

Requirements
========================
<p>scikit-optimize 0.10.1
<p>scikit-learn 1.2.2

Directory Structure
===============================
**data/: Contains the dataset files.**

<p>•	train.csv: Contains 379 optimized features, with 98 active molecules and 185 inactive molecules.
<p>•	test.csv: Contains 379 optimized features, with 25 active molecules and 46 inactive molecules.
<p>•	train-1051.csv: Contains 123 active molecules and 928 inactive molecules.

**scripts/: Contains all the scripts for running the machine learning models and analyses.**

<p>•	SVM.py
<p>•	XGBoost.py
<p>•	RF.py
<p>•	NB.py
<p>•	LR.py
<p>•	KNN.py
<p>•	ANN.py
<p>•	AdaBoost.py
<p>•	model_comparison.py
<p>•	prediction.py
<p>•	100training.py

**results/: Directory for saving output files such as metrics and predictions.**
<p>•	metrics.csv
<p>•	predictions.csv

Usage
===============================
**Finding Optimal Hyperparameters**
<p>Each algorithm script (SVM.py, XGBoost.py, RF.py, NB.py, LR.py, KNN.py, ANN.py, AdaBoost.py) is used to find the optimal hyperparameters using Bayesian optimization.
<p>Example command:

**python scripts/RF.py --input data/train.csv --output results/RF.csv --y class --X1 D316 --X2 piPC7**
<p>

**Comparing Model Performance**
<p>model_comparison.py compares the performance of different machine learning algorithms.
<p>Example command:

**python scripts/model_comparison.py --train data/train.csv --y class --X1 D316 --X2 piPC7 --metrics results/metrics.csv --predictions results/predictions.csv**
<p>
    
**Predicting Independent Test Set**
<p>prediction.py is used to predict the independent test set.
<p>Example command:

**python scripts/prediction.py --train data/train.csv --test data/test.csv --y class --X1 D316 --X2 piPC7 --metrics results/metrics.csv --predictions results/predictions.csv**
<p>
    
**Repeated Training and Testing**
<p>100training.py randomly splits the data into training and test sets, uses the RF379 model to predict the test set, and repeats the process 100 times.
<p>Example command:

**python scripts/100training.py --data data/train-1051.csv --y class --X1 D316 --X2 piPC7 --metrics results/results.csv --predictions results/predictions.csv**
<p>

Citation
========================
If you use this code in your research, please cite our study:
<p>Identification of Active Molecules Against Thrombocytopenia Through Machine Learning

Contact
==============================
For any questions or issues, please contact [ljs@swmu.edu.cn].

