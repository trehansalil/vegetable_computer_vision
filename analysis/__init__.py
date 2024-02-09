from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

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
    
class VisualHandler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
    def visual_representation(self):
        class_dirs = os.listdir(self.folder_path) # list all directories inside "train" folder
        image_dict = {} # dict to store image array(key) for every class(value)
        count_dict = {} # dict to store count of files(key) for every class(value)
        # iterate over all class_dirs
        for cls in class_dirs:
            # get list of all paths inside the subdirectory
            file_paths = glob.glob(os.path.join(self.folder_path, cls, "*"))
            # count number of files in each class and add it to count_dict
            count_dict[cls] = len(file_paths)
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