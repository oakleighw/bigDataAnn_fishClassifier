import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
import tensorflow as tf

#all class confusion matrix
def create_conf(class_df,gt, preds):
    #Show confusion matrix
    classLabels = np.unique(preds)
    the_matrix = confusion_matrix(preds, gt, labels=list(classLabels))

    names = np.unique(class_df['class'].values)
    df_cfm = pd.DataFrame(the_matrix, index = names, columns = names)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='g',cbar=False, cmap = 'crest')
    cfm_plot.set_title("Fish Classification Confusion Matrix")
    cfm_plot.set_xlabel('Predicted')
    cfm_plot.set_ylabel('True Label')


#converts list of paths to images
def path2im(p_array):
    return [resize((skimage.io.imread(x)),(64,64,3)) for x in p_array]

def process_labels(labels):
    l = tf.keras.utils.to_categorical(labels)
    return l