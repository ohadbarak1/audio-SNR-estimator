#!/usr/bin/env python
from NNmodules import SNREstimator
import numpy as np

if __name__ == '__main__':
    ntrain = 100
    nvalid = 10
    ntest = 10
    seed = 13
    np.random.seed(seed)

    train_data = np.random.rand(ntrain,40,90).astype(np.float32)
    train_labels = np.random.rand(ntrain).astype(np.float32)
    valid_data = np.random.rand(nvalid,40,90).astype(np.float32)
    valid_labels = np.random.rand(nvalid).astype(np.float32)
    test_data = np.random.rand(ntest,40,90).astype(np.float32)

    json_path = "/home/ohad/DG/par/ConvNet2D_A.json"
    model_path = "/home/ohad/DG/models/test.h5"
    SNR_trainer = SNREstimator(json_path=json_path)
    metrics_out = SNR_trainer.train_model(model_path,
                                      train_data=train_data, train_labels=train_labels,
                                      valid_data=valid_data, valid_labels=valid_labels)
    print (metrics_out.history.keys())

    loss_hist		= np.array(metrics_out.history['loss'], dtype=np.float32)
    mse_hist		= np.array(metrics_out.history['mean_absolute_error'], dtype=np.float32)
    val_loss_hist	= np.array(metrics_out.history['val_loss'], dtype=np.float32)
    val_mse_hist	= np.array(metrics_out.history['val_mean_absolute_error'], dtype=np.float32)

    print (f"mse_hist: {mse_hist}")
    print (f"val_mse_hist: {val_mse_hist}")

    pred = SNR_trainer.infer (model_path, test_data=test_data)
    print (f"predictions: {pred}")
