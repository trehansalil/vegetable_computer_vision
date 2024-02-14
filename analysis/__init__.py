from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
import os
import shutil
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

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

class ExperimentModelling:
    def __init__(self, class_weight, class_names, image_size, test_images):
        self.log_dir = "ninjacart_log"
        
        if os.path.exists(self.log_dir):
            remove_path(path=self.log_dir)
         
        self.class_weight = class_weight
        self.CLASS_NAMES = class_names
        self.image_size = image_size
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.test_images = test_images
        
    def preprocess(self, train_data, val_data, test_data):

        # Data Processing Stage with resizing and rescaling operations
        data_preprocess = tf.keras.Sequential(
            name="data_preprocess",
            layers=[
                tf.keras.layers.Resizing(self.image_size[0], self.image_size[1]),
                tf.keras.layers.Rescaling(1.0/255),
            ]
        )

        # Perform Data Processing on the train, test dataset
        self.train_ds = train_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        self.val_ds = val_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        self.test_ds = test_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    def compile_train_v1(self, 
                         model, 
                         epochs=10, 
                         ckpt_path="/tmp/checkpoint"):
        # tf.compat.v1.disable_eager_execution()
        # Implement a TensorBoard callback to log each of our model metrics for each model during the training process.
        self.model = model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, 
                                                           save_weights_only=True, 
                                                           monitor='val_loss', 
                                                           mode='min', 
                                                           save_best_only=True)
        
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             min_delta = 0.02,
                                                             patience=3, 
                                                             restore_best_weights=True)

        self.model_history = self.model.fit(self.train_ds, validation_data=self.val_ds,
                            epochs=epochs,
                            class_weight=self.class_weight,
                            callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback])

        return self.model_history

    def plot_acc_loss(self, metric='accuracy'):

        fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
        ax = axes.ravel()
        n_epochs = len(self.model_history.history[f'{metric}'])
        #accuracy graph
        ax[0].plot(range(0, n_epochs), [acc * 100 for acc in self.model_history.history[f'{metric}']], label='Train', color='b')
        ax[0].plot(range(0, n_epochs), [acc * 100 for acc in self.model_history.history[f'val_{metric}']], label='Val', color='r')
        ax[0].set_title('Accuracy vs. epoch', fontsize=15)
        ax[0].set_ylabel('Accuracy', fontsize=15)
        ax[0].set_xlabel('epoch', fontsize=15)
        ax[0].legend()

        #loss graph
        ax[1].plot(range(0, n_epochs), self.model_history.history['loss'], label='Train', color='b')
        ax[1].plot(range(0, n_epochs), self.model_history.history['val_loss'], label='Val', color='r')
        ax[1].set_title('Loss vs. epoch', fontsize=15)
        ax[1].set_ylabel('Loss', fontsize=15)
        ax[1].set_xlabel('epoch', fontsize=15)
        ax[1].legend()

        #display the graph
        plt.show()

    def grid_test_model(self):

        fig = plt.figure(1, figsize=(17, 11))
        plt.axis('off')
        n = 0
        for i in range(8):
            n += 1

            img_0 = tf.keras.utils.load_img(random.choice(self.test_images))
            img_0 = tf.keras.utils.img_to_array(img_0)
            img_0 = tf.image.resize(img_0, tuple(self.image_size))
            img_1 = tf.expand_dims(img_0, axis = 0)

            pred = self.model.predict(img_1)
            predicted_label = tf.argmax(pred, 1).numpy().item()

            for item in pred :
                item = tf.round((item*100))

                plt.subplot(2, 4, n)
                plt.axis('off')
                plt.title(f'prediction : {self.CLASS_NAMES[predicted_label]}\n\n'
                        f'{item[0]} % {self.CLASS_NAMES[0]}\n'
                        f'{item[1]} % {self.CLASS_NAMES[1]}\n'
                        f'{item[2]} % {self.CLASS_NAMES[2]}\n'
                        f'{item[3]} % {self.CLASS_NAMES[3]}\n')
                plt.imshow(img_0/255)
            plt.show()
    
    def ConfusionMatrix(self):
        # Note: This logic doesn't work with shuffled datasets
        # run model prediction and obtain probabilities
        y_pred = self.model.predict(self.test_ds)
        # get list of predicted classes by taking argmax of the probabilities(y_pred)
        predicted_categories = tf.argmax(y_pred, axis=1)
        # create list of all "y"s labels, by iterating over test dataset
        true_categories = tf.concat([y for x, y in self.test_ds], axis=0)

        print(classification_report(true_categories, predicted_categories))

        # generate confusion matrix and plot it
        cm = confusion_matrix(true_categories,predicted_categories) # last batch
        sns.heatmap(cm, 
                    annot=True, 
                    xticklabels=self.CLASS_NAMES, 
                    yticklabels=self.CLASS_NAMES, 
                    cmap="YlGnBu", 
                    fmt='g')
        plt.show()
        
    def metrics_evals(self, y_true,y_pred, X_test):
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

    def evaluate_model(self):
        # Evaluate the model
        loss, acc = self.model.evaluate(self.test_ds, verbose=2)
        print("Accuracy of the Model with Test Data: {:5.2f}%".format(100 * acc))
        print("Loss of the Model with Test Data: {:5.4f}".format(loss))   

    def plot_image(self, pred_array, true_label, img):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(pred_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ".format(self.CLASS_NAMES[predicted_label].capitalize(),
                                        100*np.max(pred_array),
                                        ),
                                        color=color)             

    def cumulative_plot_image(self):
        true_categories = tf.concat([y for x, y in self.test_ds], axis=0)
        images = tf.concat([x for x, y in self.test_ds], axis=0)
        y_pred = self.model.predict(self.test_ds)

        # Randomly sample 15 test images and plot it with their predicted labels, and the true labels.
        indices = random.sample(range(len(images)), 15)
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(4*num_cols, 2*num_rows))
        for i,index in enumerate(indices):
            plt.subplot(num_rows, num_cols, i+1)
            self.plot_image(y_pred[index], true_categories[index], images[index])

        plt.tight_layout()
        plt.show()  
        
def move_directory(source_path, destination_path):
    try:
        # Rename function can be used to move directories in Python
        os.rename(source_path, destination_path)
        print(f"Directory moved from '{source_path}' to '{destination_path}' successfully.")
    except OSError as e:
        print(f"Error: {e}")        
        
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