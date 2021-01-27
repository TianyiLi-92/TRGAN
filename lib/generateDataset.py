import h5py
import numpy as np

def read_data(filename):
    with h5py.File(filename, 'r') as hf:
        dataset_train = np.array(hf.get('train'))
        dataset_dev = np.array(hf.get('dev'))
        dataset_test = np.array(hf.get('test'))

    return dataset_train, dataset_dev, dataset_test

def make_gappy_data(dataset, mask_context, mask_gap, mask_GenOut):
    context = dataset[:,mask_context[0][0]:mask_context[0][1],mask_context[1][0]:mask_context[1][1],:].copy()

    gap = context[:,mask_GenOut[0][0]:mask_GenOut[0][1],mask_GenOut[1][0]:mask_GenOut[1][1],:].copy()

    context[:,mask_gap[0][0]:mask_gap[0][1],mask_gap[1][0]:mask_gap[1][1],:] = 0

    return context, gap

def generateDataset(filename, mask_context, mask_gap, mask_GenOut):
    dataset_train, dataset_dev, dataset_test = read_data(filename)

    context_train, gap_train = make_gappy_data(dataset_train, mask_context, mask_gap, mask_GenOut)
    context_dev, gap_dev = make_gappy_data(dataset_dev, mask_context, mask_gap, mask_GenOut)
    context_test, gap_test = make_gappy_data(dataset_test, mask_context, mask_gap, mask_GenOut)

    return context_train, gap_train, context_dev, gap_dev, context_test, gap_test