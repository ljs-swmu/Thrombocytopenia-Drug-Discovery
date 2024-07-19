#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# 定义自定义的评估指标
scoring = {
    'ACC': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1': f1_score,
    'MCC': matthews_corrcoef
}

def evaluate_model_cv(model, X, y, scaler):
    # 数据预处理
    X = scaler.fit_transform(X)
    if isinstance(model, MultinomialNB):
        # Clip to ensure non-negative values (necessary for MultinomialNB)
        X = X.clip(min=0)

    # 进行5折交叉验证，获取预测概率
    y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')
    
    # 预测类别
    y_pred_class = (y_pred_proba[:, 1] > 0.5).astype(int)
    
    # 计算性能指标
    acc = accuracy_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class, average='binary', zero_division=1)
    recall = recall_score(y, y_pred_class, average='binary')
    f1 = f1_score(y, y_pred_class, average='binary')
    mcc = matthews_corrcoef(y, y_pred_class)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y, y_pred_class).ravel()

    return acc, precision, recall, f1, mcc, y_pred_proba, tp, tn, fp, fn

def model_comparison_and_evaluation(train_file, y_col, X1_col, X2_col, metrics_file, predictions_file):
    # 读取训练数据
    train_data = pd.read_csv(train_file, sep=',', encoding='gbk')
    y_train = train_data[y_col]
    X_train = train_data.loc[:, X1_col:X2_col]

    # 定义模型和最佳参数
    models = {
        'SVM': SVC(C=0.6625513244743272, coef0=0, degree=2, gamma='auto', kernel='rbf', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=24, p=1, weights='distance'),
        'Naive Bayes': MultinomialNB(alpha=4.908351200924687e-06),
        'Random Forest': RandomForestClassifier(max_depth=5, max_features='log2', min_samples_leaf=5, min_samples_split=5, n_estimators=424),
        'ANN': MLPClassifier(activation='tanh', alpha=0.005664392362917559, hidden_layer_sizes=10, learning_rate='adaptive', solver='sgd'),
        'LR': LogisticRegression(C=0.07773697006110111, penalty='l1', solver='liblinear', max_iter=928),
        'XGBoost': XGBClassifier(colsample_bytree=0.5710321246336743, gamma=5.352688773863087, learning_rate=0.01, max_depth=10, n_estimators=98, reg_alpha=1e-09, reg_lambda=0.060015420369268865, subsample=0.9132902482766538),
        'AdaBoost': AdaBoostClassifier(learning_rate=0.11820543985658238, n_estimators=2)
    }

    # 评估每个模型
    results = {
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
    predictions = pd.DataFrame({'True Label': y_train})  # 初始化包含标签的 DataFrame

    # 数据预处理器
    scaler = StandardScaler()

    for name, model in models.items():
        acc, precision, recall, f1, mcc, y_pred_proba, tp, tn, fp, fn = evaluate_model_cv(model, X_train, y_train, scaler)

        # 存储结果
        results['Model'].append(name)
        results['ACC'].append(acc)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1'].append(f1)
        results['MCC'].append(mcc)
        results['TP'].append(tp)
        results['TN'].append(tn)
        results['FP'].append(fp)
        results['FN'].append(fn)

        # 将每个算法的预测概率作为单独的列添加到 predictions DataFrame 中
        predictions[name] = y_pred_proba[:, 1]

    # 输出性能指标到 CSV 文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(metrics_file, sep=',', index=False)

    # 输出预测概率到 CSV 文件
    predictions.to_csv(predictions_file, sep=',', index=False)

    print(f"Metrics saved to {metrics_file}")
    print(f"Predictions saved to {predictions_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Comparison and Evaluation with Cross-Validation')
    parser.add_argument('--train', help='Training file path', required=True)
    parser.add_argument('--y', help='Column to be used as target variable', required=True)
    parser.add_argument('--X1', help='Starting column of feature variables', required=True)
    parser.add_argument('--X2', help='Ending column of feature variables', required=True)
    parser.add_argument('--metrics', help='Output file path for metrics', required=True)
    parser.add_argument('--predictions', help='Output file path for predictions', required=True)
    args = parser.parse_args()

    model_comparison_and_evaluation(args.train, args.y, args.X1, args.X2, args.metrics, args.predictions)
