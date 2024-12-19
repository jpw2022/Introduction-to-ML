import numpy as np
import matplotlib.pyplot as plt
import os

def plot_acc(train_acc: list[float],
             val_acc: list[float]):
    plt.plot(train_acc, label="train accuracy")
    plt.plot(val_acc, label="validation accuracy")
    plt.xscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

def save_fig(filename: str):
    plt.legend()
    plt.savefig(os.path.join("images", filename))
