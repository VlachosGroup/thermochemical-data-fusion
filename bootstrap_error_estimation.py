"""Estimate the empirical distribution of errors by bootstrap.

Theory
------
Steps
-----
For a test set of size N and QoI being Mean Absolute Error (MAE)
1. Train a model on a training set.
2. Pick DRAW_SIZE points at random (with replacement) from a separate test set
    and evaluate the mean absolute error. This constitutes one sample of the
    QoI. Draw M such samples.
         Note
         ====
         The empirical distribution of a QoI places probability weight 1/n on
         each sample point. Thus by sampling uniformly with replacement is
         actually sampling from the empirical distribution of the QoI.
3. The sample variance of the M samples is an estimate of the true variance
    of the QoI.
4. For M ~ 10000, use Central Limit Theorem to claim that the true MAE
    converges in probability to a Normal centered on the observed MAE
    and variance as estimated in step 3.
5. Construct a confidence band for a normal distribution

Algorithm
---------
1. Load test set.
2. Load model.
3. Loop number of sample required
3. Draw sample from test set randomly with replacement.
4. Normalize X and y
4. Predict response and check mean error. Append to list of MAE samples
5. Plot.

"""
from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

from helper_files import plot_density
from lasso_fits import Model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('task',
                        help="['electronic energy', enthalpy', 'free energy']")
    parser.add_argument('--x_train', help='Path of X_train.p')
    parser.add_argument('--x_test', help='Path of X_test.p')
    parser.add_argument('--y_train', help='Path of y_train.p')
    parser.add_argument('--y_test', help='Path of y_test.p')
    parser.add_argument('--draw_size', type=int, default=1000,
                        help='size of the draw (int). Default 1000')
    parser.add_argument('-m', type=int, default=100000,
                        help='M parameter (see notes). Default 100000')
    args = parser.parse_args()

    DRAW_SIZE = args.draw_size
    M = args.m
    TASK = args.task
    X_test_path = args.x_test
    y_test_path = args.y_test

    # for labeling and color
    task_dict = {
        'electronic energy': 'ee',
        'enthalpy': 'h',
        'free energy': 'g'}
    task_label_dict = {
        'electronic energy': 'E-E',
        'enthalpy': 'E-H',
        'free energy': 'E-G'}
    task_colors = {
        'electronic energy':'#594a4e',
        'enthalpy': '#e78fb3',
        'free energy': '#6fc1a5'}

    # load testing data
    X_test = pickle.load(open(X_test_path, "rb"))
    y_test = pickle.load(open(y_test_path, "rb")).reshape(-1, 1)
    model_class = pickle.load(open(model_path, "rb"))

    mae_samples = []  # len(mae_samples) = M
    for run in range(M):
        train_idx = np.random.choice(
            a=len(y_test), size=DRAW_SIZE, replace=True)
        X_sample = X_test[train_idx, :]
        y_sample = y_test[train_idx]

        # scale X_sample and y_sample
        X_sample = model_class.X_scaler.transform(X_sample)
        y_sample = model_class.y_scaler.transform(y_sample)
        y_sample_pred = model_class.model_.predict(X_sample)
        mae_samples.append(
            float(mean_absolute_error(
                y_true=y_sample, y_pred=y_sample_pred)
                    * model_class.y_scaler.scale_))

    std_dev = np.std(mae_samples)
    CI_99_9 = 3.291 * std_dev
    mean = np.mean(mae_samples)
    # plot
    bw=0.5
    plot_density(
        mae_samples, bw=bw,
        xlabel=f'\u0394{task_label_dict[TASK]} MAE (kcal/mol)',
        color=task_colors[TASK])
    axes = plt.gca()
    plt.text(
        0.05, 0.9, f'MEAN: {mean: 0.2f}', fontsize=20,
        transform=axes.transAxes)
    plt.text(
        0.05, 0.8,
        f'99.9% CI: {CI_99_9: 0.2f}', fontsize=20,
        transform=axes.transAxes)
    plt.tight_layout()
    plt.show()
    print('SETTINGS')
    print('********')
    print(f'M: {M}')
    print(f'DRAW_SIZE: {DRAW_SIZE}')
    print(f'bw: {bw}')
    print(f'Std dev.: {std_dev}')
    print(f'Mean: {mean}')
    print(f'99.9% CI: {CI_99_9}')
