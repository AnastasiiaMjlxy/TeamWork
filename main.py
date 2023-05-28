#for refactor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_decision_boundary(svc, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.title(title)

#for refactor
def plot_svm_kernels(X, y):
    kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
    titles = ['Linear Kernel', 'RBF Kernel', 'Polynomial Kernel', 'Sigmoid Kernel']
    num_subplots = len(kernel_list)
    num_cols = 2
    num_rows = (num_subplots + num_cols - 1) // num_cols
    plt.figure(figsize=(12, 8))

    for i, kernel in enumerate(kernel_list):
        svc = svm.SVC(kernel=kernel, C=1).fit(X, y)
        subplot_idx = i + 1
        plt.subplot(num_rows, num_cols, subplot_idx)
        plot_decision_boundary(svc, X, y, titles[i])

    plt.tight_layout()
    plt.show()

#for refactor
if __name__ == "__main__":
    banknote_data = datasets.load_breast_cancer()
    X = banknote_data.data[:, :2]
    y = banknote_data.target
    plot_svm_kernels(X, y)
