import os
import configparser
import tensorflow as tf
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

config_path = os.path.join(os.getcwd(), "config_file.config")

config_parser = configparser.ConfigParser()
config_parser.read(config_path)

# All DB Inputs
data_gsheet_id = config_parser.get('config', 'data_gsheet_id')
output = config_parser.get('config', 'output_file')


# For Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def metrics_evals(y_true,y_pred, X_test):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1-r2)*(len(y_true)-1)/(len(y_true)-X_test.shape[1]-1)

    return {"MSE":mse,
            "RMSE":rmse,
            "MAE":mae,
            "R2":r2,
            "ADJ_R2": adj_r2}