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

    # model expects input in dB
    train_labels = 20*(np.log10(train_labels * 24.))
    valid_labels = 20*(np.log10(valid_labels * 24.))
    print (train_labels)
    print (valid_labels)


    json_path = "/home/ohad/DG/par/ConvNet2D_A.json"
    model_path = "/home/ohad/DG/models/test.h5"
    SNR_trainer = SNREstimator(json_path=json_path)
    metrics_out = SNR_trainer.train_model(model_path,
                                      train_data=train_data, train_labels=train_labels,
                                      valid_data=valid_data, valid_labels=valid_labels)

    for metric in metrics_out.history.keys():
        print (f"{metric}: {metrics_out.history[metric]}")

    pred = SNR_trainer.infer (model_path, test_data=test_data)
    print (f"predictions: {pred}")
