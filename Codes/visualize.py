import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import os
import torch

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

def plot_task3(opt_str: str, opt_step: int):
    val_acc = []
    alphas = []
    for file in os.listdir("./ckpt"):
        if file.endswith(".val") and (opt_str+"_0." in file):
            data = torch.load("./ckpt/"+file)
            vc = data['val_acc']
            val_acc.append(max(vc[:opt_step // data['step_per_epoch']]))
            alphas.append(data['alpha'])
    plt.figure(figsize=(5, 3))
    plt.scatter(alphas, val_acc)
    plt.title(opt_str)
    plt.ylabel("Validation accuracy")
    plt.ylim((-4, 104))
    plt.savefig(os.path.join("./images", "task3", opt_str+"_summary.pdf"))
