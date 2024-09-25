#!/usr/bin/env python
from NNmodules import SNREstimator
import argparse
import numpy as np
from os.path import dirname, join
from datetime import datetime
import hashlib
import matplotlib.pyplot as plt

def mae (x, y):
    return np.mean (np.abs(x-y))

def train_model (
        json_path,
        train_data_path,
        train_labels_path,
        valid_data_path,
        valid_labels_path,
        model_path
    ):

    train_data=np.load(train_data_path)
    train_labels=np.load(train_labels_path)
    valid_data=np.load(valid_data_path)
    valid_labels=np.load(valid_labels_path)

    SNR_trainer = SNREstimator(json_path=json_path)
    metrics_out = SNR_trainer.train_model(model_path,
                                          train_data=train_data, train_labels=train_labels,
                                          valid_data=valid_data, valid_labels=valid_labels)
    return metrics_out

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
    ap.add_argument('train_data', type=str, help='training data numpy file path')
    ap.add_argument('train_labels', type=str, help='training labels numpy file path')
    ap.add_argument('valid_data', type=str, help='validation data numpy file path')
    ap.add_argument('valid_labels', type=str, help='validation labels numpy file path')
    ap.add_argument('model_pre', type=str, help='output model filename prefix')
    args, unparsed = ap.parse_known_args()

    date_string = f'{datetime.now():%Y%m%d_%H%M%S}'
    hash = hashlib.shake_128(date_string.encode("ascii")).hexdigest(8)
    model_dir = dirname(args.train_data)
    model_dir = join (model_dir, args.model_pre+'_'+hash)
    model_path = join (model_dir, 'model.h5')

 
    metrics = train_model (
        args.json_params,
        args.train_data,
        args.train_labels,
        args.valid_data,
        args.valid_labels,
        model_path
    )

    valid_labels = np.load(args.valid_labels)
    pred = infer_model(args.json_params, args.valid_data, model_path)
    
    valid_pred_loss = mae (valid_labels, pred)
    print (f"MAE loss from inference on validation data = {valid_pred_loss}")
    
    for metric in metrics.history.keys():
        print (f"{metric}: {metrics.history[metric]}")
    
    plot_file = join (model_dir, "metrics_output.png")

    fig, [ax0, ax1] = plt.subplots (2, 1, figsize=(16,12))

    ax0.plot (metrics.history['loss'], linewidth=3)
    ax0.plot (metrics.history['val_loss'], linewidth=3)
    ax0.grid(True, alpha=0.3)
    #ax0.set_ylim (ymin=0, ymax=20)
    ax0.set_ylabel('loss', fontsize=10)
    ax0.set_xlabel ('epochs', fontsize=10)
    ax0.autoscale(enable=True, axis='x', tight=True)

    ax1.plot (metrics.history['mean_absolute_error'], linewidth=3)
    ax1.plot (metrics.history['val_mean_absolute_error'], linewidth=3)
    ax1.grid(True, alpha=0.3)
    #ax1.set_ylim (ymin=0, ymax=20)
    ax1.set_ylabel('MAE loss', fontsize=10)
    ax1.set_xlabel ('epochs', fontsize=10)
    ax1.autoscale(enable=True, axis='x', tight=True)
    print('saving figures to {}'.format(plot_file))
    plt.savefig(plot_file, dpi=200)
    plt.close()



