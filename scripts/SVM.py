#!/usr/bin/env python

import argparse
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Integer, Real  # 确保导入这些类
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, matthews_corrcoef
import multiprocessing  # 用于并行计算

def main(input_file, output_file, y_col, X1_col, X2_col):
    # 读取数据
    data = pd.read_csv(input_file, sep=',', encoding='gbk')
    y = data[y_col]  # 将指定的列作为目标变量 y
    X = data.loc[:, X1_col:X2_col]  # 将指定的列范围作为特征变量 X

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 定义参数空间
    param_space = {
        'C': Real(1e-6, 1e3, prior='log-uniform'), # C 的范围缩小到 0.1 到 10
        'coef0': Real(0, 10),  # coef0 的范围为 0 到 10
        'degree': Integer(2, 10),  # degree 的范围为 2 到 10
        'gamma': ['auto'],  # gamma 固定为 'auto'
        'kernel': ['rbf'],  # 核函数固定为 'rbf'
        'probability': [True]  # 确定为计算概率
    }

    # 初始化默认模型
    default_model = SVC(kernel='rbf', probability=True, random_state=42)

    # 使用默认参数的模型进行交叉验证预测
    y_pred_default = cross_val_predict(default_model, X_scaled, y, cv=5)  # 减少折数

    # 计算默认模型的综合交叉验证准确率和MCC
    default_accuracy = accuracy_score(y, y_pred_default)
    default_mcc = matthews_corrcoef(y, y_pred_default)

    # 打印默认模型的结果
    print("Default model cross-validation accuracy: ", default_accuracy)
    print("Default model cross-validation MCC: ", default_mcc)

    # 初始化模型和贝叶斯优化
    bayes_search = BayesSearchCV(
        estimator=SVC(random_state=42),
        search_spaces=param_space,
        n_iter=20,  # 减少迭代次数以加快优化速度
        cv=5,  # 减少交叉验证折数
        n_jobs=multiprocessing.cpu_count(),  # 使用所有可用的CPU进行并行计算
        random_state=42,
        refit='accuracy'  # 指定用于指导优化过程的评分标准
    )

    # 执行贝叶斯优化
    bayes_search.fit(X_scaled, y)

    # 查看最佳参数和最佳得分
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best cross-validation accuracy score: ", bayes_search.best_score_)

    # 使用最佳参数训练模型，并在整个数据集上进行交叉验证预测
    best_model = SVC(**bayes_search.best_params_, random_state=42)
    y_pred = cross_val_predict(best_model, X_scaled, y, cv=5)  # 减少交叉验证折数

    # 计算综合的ACC和MCC
    overall_accuracy = accuracy_score(y, y_pred)
    overall_mcc = matthews_corrcoef(y, y_pred)

    print("Overall cross-validation accuracy: ", overall_accuracy)
    print("Overall cross-validation MCC: ", overall_mcc)

    # 保存结果到CSV
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
    parser = argparse.ArgumentParser(description='Support Vector Classifier with Bayesian Optimization')
    parser.add_argument('--input', help='input file path', required=True)
    parser.add_argument('--output', help='output file path', required=True)
    parser.add_argument('--y', help='column to be used as target variable', required=True)
    parser.add_argument('--X1', help='starting column of feature variables', required=True)
    parser.add_argument('--X2', help='ending column of feature variables', required=True)
    args = parser.parse_args()
    main(args.input, args.output, args.y, args.X1, args.X2)