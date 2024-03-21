#!/usr/bin/env python
from NNmodules import SNREstimator
import argparse
import numpy as np
from os.path import dirname, join, basename
import matplotlib.pyplot as plt

def mae (x, y):
    return np.mean (np.abs(x.flatten()-y.flatten()))

def infer_model (
        json_path,
        test_data_path,
        model_path
    ):

    test_data = np.load(test_data_path)
    SNR_trainer = SNREstimator(json_path=json_path)
    pred = SNR_trainer.infer(model_path, test_data)

    return pred

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('json_params', type=str, help='json input parameters file')
    ap.add_argument('test_data', type=str, help='test data numpy file path')
    ap.add_argument('test_labels', type=str, help='test labels numpy file path')
    ap.add_argument('model_path', type=str, help='input model filename')
    args, unparsed = ap.parse_known_args()

    pred = infer_model(args.json_params, args.test_data, args.model_path)
    test_labels = np.load (args.test_labels)
    pred_loss = mae (test_labels, pred)
    print (f"MAE dB loss from inference on test data = {pred_loss}")
    
    delta_pred = pred.flatten()-test_labels.flatten()
    delta_abs = np.abs(delta_pred)
    min_delta_pred = 0
    max_delta_pred = 15

    delta_SNR=np.arange(min_delta_pred, max_delta_pred, float((max_delta_pred-min_delta_pred))/100)
    ROC = [delta_abs[delta_abs < i].size / delta_abs.size for i in delta_SNR]

    plot_file = join (dirname(args.model_path), basename(args.test_data)+"_"+"inference_delta.png")

    fig, [ax0, ax1] = plt.subplots (2, 1, figsize=(8,12))

    dB_bins = np.arange(-max_delta_pred,max_delta_pred,1)
    ax0.hist(delta_pred, bins=dB_bins, density=True, color='skyblue', edgecolor='black')
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim (ymin=0, ymax=1)
    ax0.set_ylabel('pct of samples', fontsize=18)
    ax0.set_xlabel ('prediction delta_SNR', fontsize=18)
    ax0.set_xlim(-max_delta_pred, max_delta_pred)

    ax1.plot(delta_SNR, ROC)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim (ymin=0, ymax=1.1)
    ax1.set_ylabel('pct in range', fontsize=18)
    ax1.set_xlabel ('prediction delta_SNR', fontsize=18)


    print('saving figures to {}'.format(plot_file))
    plt.savefig(plot_file, dpi=200)
    plt.close()



