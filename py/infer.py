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
    plot_file = join (dirname(args.model_path), basename(args.test_data)+"_"+"inference_delta.png")

    fig, ax0 = plt.subplots (1, 1, figsize=(8,6))


    ax0.hist(delta_pred, bins=24, color='skyblue', edgecolor='black')
    ax0.grid(True, alpha=0.3)
    #ax0.set_ylim (ymin=0, ymax=20)
    ax0.set_ylabel('number of samples', fontsize=10)
    ax0.set_xlabel ('prediction delta_SNR', fontsize=10)
    ax0.set_xlim(-15, 15)
    print('saving figures to {}'.format(plot_file))
    plt.savefig(plot_file, dpi=200)
    plt.close()



