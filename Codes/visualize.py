import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import os

def plot_acc(train_acc: list[float],
             val_acc: list[float]):
    plt.plot(train_acc, label="training")
    plt.plot(val_acc, label="validation")
    plt.xscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

def add_title(title: str):
    plt.title(title)

def save_fig(filename: str):
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig(os.path.join("./images", filename))
