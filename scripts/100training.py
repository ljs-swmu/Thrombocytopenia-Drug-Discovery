#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    # 数据预处理
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if isinstance(model, MultinomialNB):
        # Clip to ensure non-negative values (necessary for MultinomialNB)
        X_train = X_train.clip(min=0)
        X_test = X_test.clip(min=0)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    y_pred_class = (y_pred_proba > 0.5).astype(int) if y_pred_proba is not None else model.predict(X_test)

    # 计算性能指标
    acc = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, average='binary', zero_division=1)
    recall = recall_score(y_test, y_pred_class, average='binary')
    f1 = f1_score(y_test, y_pred_class, average='binary')
    mcc = matthews_corrcoef(y_test, y_pred_class)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()

    return acc, precision, recall, f1, mcc, tp, tn, fp, fn, y_pred_proba

def model_comparison_and_evaluation(data_file, y_col, X1_col, X2_col, metrics_file, predictions_file):
    # 读取数据
    data = pd.read_csv(data_file, sep=',', encoding='gbk')
    y = data[y_col]
    X = data.loc[:, X1_col:X2_col]

    # 定义模型和最佳参数
    models = {
        'Random Forest': RandomForestClassifier(max_depth=5, max_features='log2', min_samples_leaf=5, min_samples_split=5, n_estimators=424)
    }

    # 评估每个模型
    metrics_results = {
        'Iteration': [],
        'Model': [],
        'ACC': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'MCC': [],
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': []
    }

    all_predictions = []

    # 数据预处理器
    scaler = StandardScaler()

    for iteration in range(1, 101):
        print(f'Iteration: {iteration}')

        # 打乱数据
        data_shuffled = data.sample(frac=1, random_state=iteration).reset_index(drop=True)
        y = data_shuffled[y_col]
        X = data_shuffled.loc[:, X1_col:X2_col]

        # 随机打乱并划分数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=iteration)

        for name, model in models.items():
            acc, precision, recall, f1, mcc, tp, tn, fp, fn, y_pred_proba = evaluate_model(model, X_train, y_train, X_test, y_test, scaler)

            # 存储结果
            metrics_results['Iteration'].append(iteration)
            metrics_results['Model'].append(name)
            metrics_results['ACC'].append(acc)
            metrics_results['Precision'].append(precision)
            metrics_results['Recall'].append(recall)
            metrics_results['F1'].append(f1)
            metrics_results['MCC'].append(mcc)
            metrics_results['TP'].append(tp)
            metrics_results['TN'].append(tn)
            metrics_results['FP'].append(fp)
            metrics_results['FN'].append(fn)

            # 将预测概率加入到列表中
            if y_pred_proba is not None:
                predictions_df = pd.DataFrame({
                    'Iteration': iteration,
                    'Model': name,
                    'True Label': y_test,
                    'Predicted Proba': y_pred_proba
                })
                all_predictions.append(predictions_df)

    # 输出所有迭代的性能指标到 CSV 文件
    results_df = pd.DataFrame(metrics_results)
    results_df.to_csv(metrics_file, sep=',', index=False)

    # 合并所有预测结果并输出到单个CSV文件
    all_predictions_df = pd.concat(all_predictions, axis=0)
    all_predictions_df.to_csv(predictions_file, sep=',', index=False)

    print(f"Metrics saved to {metrics_file}")
    print(f"All predictions saved to {predictions_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Comparison and Evaluation with Repeated Random Splits')
    parser.add_argument('--data', help='Data file path', required=True)
    parser.add_argument('--y', help='Column to be used as target variable', required=True)
    parser.add_argument('--X1', help='Starting column of feature variables', required=True)
    parser.add_argument('--X2', help='Ending column of feature variables', required=True)
    parser.add_argument('--metrics', help='Output file path for metrics', required=True)
    parser.add_argument('--predictions', help='Output file path for predictions', required=True)
    args = parser.parse_args()

    model_comparison_and_evaluation(args.data, args.y, args.X1, args.X2, args.metrics, args.predictions)
