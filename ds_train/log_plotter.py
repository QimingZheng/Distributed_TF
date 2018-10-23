import matplotlib.pyplot as plt
import numpy as np

def read_train_log(filename):
    time_stamp = []
    cross_entro = []
    with open(filename) as file:
        text = file.readlines()
        total_loss= 0.0
        for i in range(len(text)):
            text[i] = text[i].split()
            total_loss += float(text[i][1])
            if int(text[i][0])%2180==0 and i:
                time_stamp.append(i/2180)
                cross_entro.append(total_loss/2180)
                total_loss = 0.0
    return time_stamp, cross_entro

def read_infer_log(filename):
    time_stamp = []
    cross_entro = []
    with open(filename) as file:
        text = file.readlines()
        for i in range(len(text)):
            text[i] = text[i].split()
            loss = float(text[i][1])
            time_stamp.append(int(text[i][0]))
            cross_entro.append(loss)
    return time_stamp, cross_entro

def plot(T_time_stamp, T_sequence, E_time_stamp, E_sequence, I_time_stamp, I_sequence):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(T_time_stamp, T_sequence, 'ro-', label="train-curve")
    ax1.plot(E_time_stamp, E_sequence, 'go-', label="validation-curve")
    ax1.set_xlabel("Passes")
    ax1.set_ylabel("Cross Entropy Loss")
    plt.legend()
    ax2 = ax1.twinx()
    ax2.plot(I_time_stamp, I_sequence, 'b^-', label="inference-curve")
    plt.grid()
    plt.legend()
    ax2.set_ylabel("BLEU Score")
    plt.savefig("batch-loss.png")
    return

T_time_stamp, T_sequence = read_train_log("train.log")
E_time_stamp, E_sequence = read_infer_log("eval.log")
I_time_stamp, I_sequence = read_infer_log("infer.log")
plot(T_time_stamp, T_sequence, E_time_stamp, E_sequence, I_time_stamp, I_sequence)
