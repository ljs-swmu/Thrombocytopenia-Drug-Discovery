#!/usr/bin/env python

import argparse
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, matthews_corrcoef
import multiprocessing

def main(input_file, output_file, y_col, X1_col, X2_col):
    # Read data
    data = pd.read_csv(input_file, sep=',', encoding='gbk')
    y = data[y_col]  # Target variable
    X = data.loc[:, X1_col:X2_col]  # Features

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Parameter space for DecisionTreeClassifier
    param_space = {
        'max_depth': Integer(3, 15),  # Maximum depth of the tree
        'min_samples_split': Integer(2, 20),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': Integer(1, 10),  # Minimum number of samples required to be at a leaf node
        'criterion': Categorical(['gini', 'entropy']),  # Split criterion
    }

    # Initialize default model (DecisionTreeClassifier)
    default_model = DecisionTreeClassifier(random_state=42)

    # Default model cross-validation prediction
    y_pred_default = cross_val_predict(default_model, X_scaled, y, cv=5)

    # Compute default model metrics
    default_accuracy = accuracy_score(y, y_pred_default)
    default_mcc = matthews_corrcoef(y, y_pred_default)

    # Print default model results
    print("Default model cross-validation accuracy: ", default_accuracy)
    print("Default model cross-validation MCC: ", default_mcc)

    # Initialize model and Bayesian optimization (DecisionTreeClassifier)
    bayes_search = BayesSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        search_spaces=param_space,
        n_iter=20,
        cv=5,
        n_jobs=multiprocessing.cpu_count(),
        random_state=42,
        refit='accuracy'
    )

    # Perform Bayesian optimization
    bayes_search.fit(X_scaled, y)

    # Print best parameters and best score
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best cross-validation accuracy score: ", bayes_search.best_score_)

    # Train model with best parameters and perform cross-validation prediction
    best_model = DecisionTreeClassifier(random_state=42, **bayes_search.best_params_)
    y_pred = cross_val_predict(best_model, X_scaled, y, cv=5)

    # Compute overall accuracy and MCC
    overall_accuracy = accuracy_score(y, y_pred)
    overall_mcc = matthews_corrcoef(y, y_pred)

    print("Overall cross-validation accuracy: ", overall_accuracy)
    print("Overall cross-validation MCC: ", overall_mcc)

    # Save results to CSV
    results = {
        'Best Parameters': [bayes_search.best_params_],
        'Best CV Accuracy': [bayes_search.best_score_],
        'Default CV Accuracy': [default_accuracy],
        'Default CV MCC': [default_mcc],
        'Overall CV Accuracy': [overall_accuracy],
        'Overall CV MCC': [overall_mcc]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decision Tree Classifier with Bayesian Optimization')
    parser.add_argument('--input', help='input file path', required=True)
    parser.add_argument('--output', help='output file path', required=True)
    parser.add_argument('--y', help='column to be used as target variable', required=True)
    parser.add_argument('--X1', help='starting column of feature variables', required=True)
    parser.add_argument('--X2', help='ending column of feature variables', required=True)
    args = parser.parse_args()
    main(args.input, args.output, args.y, args.X1, args.X2)
