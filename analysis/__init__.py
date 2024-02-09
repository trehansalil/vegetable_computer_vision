from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
import os
import shutil
import random
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(train_data, val_data, test_data, target_height=227, target_width=227):

    # Data Processing Stage with resizing and rescaling operations
    data_preprocess = tf.keras.Sequential(
        name="data_preprocess",
        layers=[
            tf.keras.layers.Resizing(target_height, target_width),
            tf.keras.layers.Rescaling(1.0/255),
        ]
    )

    # Perform Data Processing on the train, test dataset
    train_ds = train_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

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

def remove_path(path):
    # Check if the file exists before attempting to delete it
    if os.path.exists(path):
        try:
            # Delete the file
            os.remove(path)
            print(f"{path} has been deleted.")
        except:
            shutil.rmtree(path)
            print(f"{path} has been deleted.")
    else:
        print(f"The path {path} does not exist.")      
        
class VisualHandler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
    def visual_representation_by_class(self):
        class_dirs = os.listdir(self.folder_path) # list all directories inside "train" folder
        image_dict = {} # dict to store image array(key) for every class(value)
        self.count_dict = {} # dict to store count of files(key) for every class(value)
        # iterate over all class_dirs
        for cls in class_dirs:
            # get list of all paths inside the subdirectory
            file_paths = glob.glob(os.path.join(self.folder_path, cls, "*"))
            # count number of files in each class and add it to count_dict
            self.count_dict[cls] = len(file_paths)
            # select random item from list of image paths
            image_path = random.choice(file_paths)
            # load image using keras utility function and save it in image_dict
            image_dict[cls] = tf.keras.utils.load_img(image_path)
            
        ## Viz Random Sample from each class
        plt.figure(figsize=(15, 12))
        # iterate over dictionary items (class label, image array)
        for i, (cls,img) in enumerate(image_dict.items()):
            # create a subplot axis
            ax = plt.subplot(3, 4, i + 1)
            # plot each image
            plt.imshow(img)
            # set "class name" along with "image size" as title
            plt.title(f'{cls}, {img.size}')
            plt.axis("off") 
                    
    def class_count(self):        
        ## Let's now Plot the Data Distribution of Training Data across Classes
        df_count_train = pd.DataFrame({
            "class": self.count_dict.keys(),     # keys of count_dict are class labels
            "count": self.count_dict.values(),   # value of count_dict contain counts of each class
        })
        print("Count of training samples per class:\n", df_count_train)

        # draw a bar plot using pandas in-built plotting function
        df_count_train.plot.bar(x='class', y='count', title="Training Data Count per class")
        plt.show()                  