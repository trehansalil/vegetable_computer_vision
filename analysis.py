import numpy as np
import pandas as pd
import requests
import subprocess
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

import keras

from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# For Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
import gdown

data_gsheet_id = "1clZX-lV_MLxKHSyeyTheX5OCQtNCUcqT"
output = 'ninjacart_data.zip'
gdown.download(id=data_gsheet_id, output=output)