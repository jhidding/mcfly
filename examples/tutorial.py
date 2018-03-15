
# coding: utf-8

# # Tutorial PAMAP2 with mcfly

# The goal of this tutorial is to get you familiar with training Neural Networks for time series using mcfly. At the end of the tutorial, you will have compared several Neural Network architectures you know how to train the best performing network.
# 
# As an example dataset we use the publicly available [PAMAP2 dataset](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring). It contains time series data from movement sensors worn by nine individuals. The data is labelled with the activity types that these individuals did and the aim is to train and evaluate a *classifier*.
# 
# Before you can start, please make sure you install mcfly (see the [mcfly installation page](https://github.com/NLeSC/mcfly)).

# ## Import required Python modules

# In[1]:


import sys
import os
import numpy as np
import pandas as pd
# mcfly
from mcfly import modelgen, find_architecture, storage
from keras.models import load_model
np.random.seed(2)


# In[2]:


sys.path.insert(0, os.path.abspath('../..'))
from utils import tutorial_pamap2


# ## Download data pre-procesed data

# We have created a function for you to fetch the preprocessed data from https://zenodo.org/record/834467. Please specify the `directory_to_extract_to` in the code below and then execute the cell. This will download the preprocessed data into the directory in the `data` subdirectory. The output of the function is the path where the preprocessed data was stored.

# In[3]:


# Specify in which directory you want to store the data:
directory_to_extract_to = '.'


# In[4]:


data_path = tutorial_pamap2.download_preprocessed_data(directory_to_extract_to)


# ## A bit about the data

# The [PAMAP2 dataset](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) contains data from three movement sensors worn by nine test subjects. These subjects performed a protocol of several activities.
# 
# The data originates from three sensors (on the hand, ankle and chest) and from each of the sensors we have three channels (acceleration on x, y and z axes). This gives us, for each time step, 9 values. The data is recorded on 100 Hz.
# 
# The preprocessed data is split into smaller segments with a window of 512 time steps, corresponding to 5.12 seconds. We only include segments that completely fall into one activity period: the activity is the *label* of the segment.
# 
# The goal of classification is to assign an activity label to an previously unseen segment.

# ## Load the pre-processed data

# Load the preprocessed data as stored in Numpy-files. Please note that the data has already been split up in a training (training), validation (val), and test subsets. It is common practice to call the input data X and the labels y.

# In[5]:


X_train, y_train_binary, X_val, y_val_binary, X_test, y_test_binary, labels = tutorial_pamap2.load_data(data_path)


# Data X and labels y are of type Numpy array. In the cell below we inspect the shape of the data. As you can see the shape of X is expressed as a Python tuple containing: the number of samples, length of the time series, and the number of channels for each sample. Similarly, the shape of y is represents the number of samples and the number of classes (unique labels). Note that y has the format of a binary array where only the correct class for each sample is assigned a 1. This is called one-hot-encoding.

# In[6]:


print('x shape:', X_train.shape)
print('y shape:', y_train_binary.shape)


# The data is split between train test and validation.

# In[7]:


print('train set size:', X_train.shape[0])
print('validation set size:', X_val.shape[0])
print('test set size:', X_test.shape[0])


# Let's have a look at the distribution of the labels:

# In[8]:


frequencies = y_train_binary.mean(axis=0)
frequencies_df = pd.DataFrame(frequencies, index=labels, columns=['frequency'])
frequencies_df


# ### *Question 1: How many channels does this dataset have?*
# ### *Question 2: What is the least common activity label in this dataset?*
# 
#     

# ## Generate models

# First step in the development of any deep learning model is to create a model architecture. As we do not know what architecture is best for our data we will create a set of random models to investigate which architecture is most suitable for our data and classification task. This process, creating random models, checking how good they are and then selecting the best one is called a 'random search'. A random search is considered to be the most robust approach to finding a good model. You will need to specificy how many models you want to create with argument 'number_of_models'. See for a full overview of the optional arguments the function documentation of modelgen.generate_models by running `modelgen.generate_models?`.
# 
# ##### What number of models to select?
# This number differs per dataset. More models will give better results but it will take longer to evaluate them. For the purpose of this tutorial we recommend trying only 2 models to begin with. If you have enough time you can try a larger number of models, e.g. 10 or 20 models. Because mcfly uses random search, you will get better results when using more models.

# In[9]:


num_classes = y_train_binary.shape[1]

models = modelgen.generate_models(X_train.shape,
                                  number_of_classes=num_classes,
                                  number_of_models = 4)


# # Inspect the models
# We can have a look at the models that were generated. The layers are shown as table rows. Most common layer types are 'Convolution' and 'LSTM' and 'Dense'. For more information see the [mcfly user manual](https://github.com/NLeSC/mcfly/wiki/User-manual) and the [tutorial cheat sheet](https://github.com/NLeSC/mcfly-tutorial/blob/master/cheatsheet.md). The summary also shows the data shape of each layer output and the number of parameters that are trained within this layer.

# In[10]:


models_to_print = range(len(models))
for i, item in enumerate(models):
    if i in models_to_print:
        model, params, model_types = item
        print("-------------------------------------------------------------------------------------------------------")
        print("Model " + str(i))
        print(" ")
        print("Hyperparameters:")
        print(params)
        print(" ")
        print("Model description:")
        model.summary()
        print(" ")
        print("Model type:")
        print(model_types)
        print(" ")


# ### *Question 3: Can you guess what hyperparameter 'learning rate' stands for?*

# ## Compare models
# Now that the model architectures have been generated it is time to compare the models by training them on a subset of the training data and evaluating the models on the validation subset. This will help us to choose the best candidate model. The performance results for the models are stored in a json file, which we will visually inspect later on.

# In[11]:


# Define directory where the results, e.g. json file, will be stored
resultpath = os.path.join(directory_to_extract_to, 'data/models')
if not os.path.exists(resultpath):
        os.makedirs(resultpath)


# We are now going to train each of the models that we generated. On the one hand we want to train them as quickly as possible in order to be able to pick the best one as soon as possible. On the other hand we have to train each model long enough to get a good impression of its potential.
# 
# We can influence the train time by adjusting the number of data samples that are used. This can be set with the argument 'subset_size'. We can also adjust the number of times the subset is iterated over. This number is called an epoch. We recommend to start with no more than 5 epochs and a maximum subset size of 300. You can experiment with these numbers.

# In[12]:


from noodles import run_process
from mcfly.storage import serial_registry

def run(wf):
    return run_process(wf, n_processes=4, registry=serial_registry)

outputfile = os.path.join(resultpath, 'modelcomparison.json')
histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train_binary,
                                                                           X_val, y_val_binary,
                                                                           models,nr_epochs=5,
                                                                           subset_size=300,
                                                                           verbose=True,
                                                                           outputfile=outputfile, use_noodles=True)
print('Details of the training process were stored in ',outputfile)

