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
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix


#commented-out code already ran to initialise data text file
############################################################
#Create text file with labels

#imPath_df = pd.DataFrame(columns=["path","class"])
############################################################

#Generate pandas df with paths and labels

# dfInd = 0
# for i in range(0,len(classes)):
#     imList = os.listdir(p+"/"+classes[i])
#     for j in range(0,len(imList)):
#         imPath_df.loc[dfInd] = [p+ "/" + classes[i] + "/" + str(imList[j])] + [i]
#         dfInd += 1

############################################################

# For generating text file
# with open('fish_labels.txt', 'w') as f:
#     for i in range(0,len(classes)):
#         imList = os.listdir(p+"/"+classes[i])
#         for j in range(0,len(imList)):
#             f.write(p+ "/" + classes[i] + "/" + str(imList[j]) + " " + str(i) + "\n")

############################################################


#creates image path and associate class dataframes (For reading text file)
def create_dataframes(class_text_file):
    full_data = pd.read_csv(class_text_file,delim_whitespace=True, header=None, names = ["imP","label"])


    debug_data = pd.DataFrame(columns=["imP","label"]) #smaller dataset for development

    for i in range(0,5):
        new_imPath_df = full_data[full_data["label"]== i].iloc[:100]
        debug_data = pd.concat([debug_data,new_imPath_df])
    
    return full_data, debug_data




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

#get per class confusion matrix and calculate metrics
def get_metrics(GT,RESULTS):

    mcm = multilabel_confusion_matrix(GT, RESULTS, labels=[0,1,2,3,4])
    Pre = []
    Rec = []
    Spec = []
    F1 = []
    CohenK = []

    #calculate per class
    plt.figure(figsize = (20,20))
    plt.tight_layout()
    for i, conf in enumerate(mcm):
        tn, fp, fn, tp = conf.ravel() 
        pre = tp/(tp + fp)
        rec = tp/(tp + fn)
        spec = tn/(tn + fp)
        f1 = (2*(pre*rec))/(pre+rec)
        coK = 2*(tp*tn - fp*fn) / ((tp+fp) * (fp+tn) + (tp+fn) *(fn+tn))

        Pre.append(pre)
        Rec.append(rec)
        F1.append(f1)
        CohenK.append(coK)
        Spec.append(spec)
    
    accuracy = accuracy_score(GT,RESULTS)
    avPre = np.mean(Pre)
    avRec = np.mean(Rec)
    avSpec = np.mean(Spec)
    avF1 = np.mean(F1)
    avCohenK = np.mean(CohenK)



    #do per class confusion matrix

    #add gridsearch
    #gradcam?
    #normalise?
    #make table
    print("Accuracy:",accuracy)
    print("Average Precision:",avPre)
    print("Average Recall/Sensitivity:",avRec)
    print("Average Specificity:",avSpec)
    print("Average F1 Score",avF1)
    print("Average Cohen K", avCohenK)


#data generator for configuring batch size
class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, x_data, y_data, batch_size):
    self.x, self.y = x_data, y_data
    self.batch_size = batch_size
    self.num_batches = np.ceil(len(x_data) / batch_size)
    self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

  def __len__(self):
    return len(self.batch_idx)

  def __getitem__(self, idx):
    batch_x = self.x[self.batch_idx[idx]]
    batch_y = self.y[self.batch_idx[idx]]
    return batch_x, batch_y