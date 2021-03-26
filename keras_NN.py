import keras
from tensorflow import keras
from keras.models import Sequential
from keras_NN import *
from Plot_Quantities import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from atlasify import atlasify
import pickle

#From Adversarial code
# Import(s)
import os
import h5py

# Set Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Keras import(s)
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam

# Local import(s) -- after Keras backend is set
import ops
import plots
import layers

#Standard neural network model
def define_model(input_size):
    
    model = Sequential()
    model.add(Dense(64, input_dim = input_size, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #Compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall])
    
    return model

def report_NN(clf, hist_clf, y_pred, x, y, m, s_fac_trim):

    #Round predictions
    y_pred_rounded = np.rint(y_pred)

    #Evaluate the naive model metrics
    loss, accuracy, precision, recall= clf.evaluate(x, y)
    print('Accuracy: %.2f' % (accuracy*100))
    print('Loss: %.2f' % (loss))
    print('Precision: %.2f' % (precision))
    print('Recall: %.2f' % (recall))
    print('F1: ', metrics.f1_score(y, y_pred_rounded))

    #Scaling constant
    SBSF = 24.838677879576565

    #Sensitivity for different cuts
    ss = []
    t = np.linspace(0.1,0.9,20,endpoint=True)
    for i in t:
        true_pos, false_pos, true_neg, false_neg = calc_Truths(y_pred, y, m, threshold = i)
        eff_sig = len(true_pos)/(len(true_pos)+len(false_neg))
        eff_bkg = len(false_pos)/(len(false_pos)+len(true_neg))
        sensitivity = Sensitivity(true_pos, false_pos, s_fac_trim, SBSF)
        ss.append(sensitivity)
        print('Sensitivity on cut {}: '.format(i), sensitivity, ' with efsig: ', eff_sig, ' with rj: ', eff_bkg)
    print('Best at: ', t[ss.index(max(ss))])

    #Plot sensitivities
    plt.plot(t, ss, label = 'Sensitivity on Cut')
    plt.xlabel('Prediction Cut')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.show()

    #Print graphs for the best cut
    true_pos, false_pos, true_neg, false_neg = calc_Truths(y_pred, y, m, threshold =t[ss.index(max(ss))])
    eff_sig = len(true_pos)/(len(true_pos)+len(false_neg))
    eff_bkg = len(false_pos)/(len(false_pos)+len(true_neg))
    sensitivity = Sensitivity(true_pos, false_pos, s_fac_trim, SBSF)
    print('Best effsig: ', eff_sig)
    print('Best rj: ', eff_bkg)
    print('Best Sensitivity: ', sensitivity)
    plot_Sculpting(true_pos, false_pos, true_neg, false_neg)
    distribution(y, y_pred, xlabel="Classifier Output")
    plt.show()
    fit_Exp(false_pos)

    #Plot metrics
    plot_ROC(y, y_pred)
    plot_PR(y, y_pred)
    plot_Precision(hist_clf)
    plot_Recall(hist_clf)
    plot_Loss(hist_clf)

#Extra model metrics
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall