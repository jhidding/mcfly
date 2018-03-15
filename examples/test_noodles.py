from mcfly import find_architecture
import numpy as np
from keras.utils.np_utils import to_categorical
import os
import unittest
import logging

import noodles
from mcfly.storage import serial_registry


def test_find_best_architecture():
    """Find_best_architecture should return a single model, parameters,
    type and valid knn accuracy."""
    np.random.seed(123)
    num_timesteps = 100
    num_channels = 2
    num_samples_train = 5
    num_samples_val = 3
    X_train = np.random.rand(
        num_samples_train,
        num_timesteps,
        num_channels)
    y_train = to_categorical(np.array([0, 0, 1, 1, 1]))
    X_val = np.random.rand(num_samples_val, num_timesteps, num_channels)
    y_val = to_categorical(np.array([0, 1, 1]))

    def run(wf):
        return noodles.run_process(
            wf, n_processes=2, registry=serial_registry, verbose=True)

    best_model, best_params, best_model_type, knn_acc = \
        find_architecture.find_best_architecture(
            X_train, y_train, X_val, y_val, verbose=True, subset_size=10,
            number_of_models=1, nr_epochs=1, use_noodles=True)
    assert hasattr(best_model, 'fit')
    assert best_params is not None
    assert best_model_type is not None
    assert 1 >= knn_acc >= 0


def test_train_models_on_samples_empty():
    np.random.seed(123)
    num_timesteps = 100
    num_channels = 2
    num_samples_train = 5
    num_samples_val = 3
    X_train = np.random.rand(
        num_samples_train,
        num_timesteps,
        num_channels)
    y_train = to_categorical(np.array([0, 0, 1, 1, 1]))
    X_val = np.random.rand(num_samples_val, num_timesteps, num_channels)
    y_val = to_categorical(np.array([0, 1, 1]))

    histories, val_metrics, val_losses = \
        find_architecture.train_models_on_samples(
            X_train, y_train, X_val, y_val, [],
            nr_epochs=1, subset_size=10, verbose=False,
            outputfile=None, early_stopping=False,
            batch_size=20, metric='accuracy', use_noodles=True)
    assert len(histories) == 0


if __name__ == "__main__":
    test_find_best_architecture()
    test_train_models_on_samples_empty()
