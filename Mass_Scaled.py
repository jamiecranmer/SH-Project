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

#Total data set provided (signal/backgorund event numberss)
samples = [447834, 2111964]
print('Raw: ', samples)

#Retrieve transformed data
with open("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/mass_scaled_data.data", 'rb') as f:
    # read the data as binary data stream
    a = pickle.load(f)
df = pd.DataFrame(a)#895668

#Standardise data
for i in range(df.shape[1]-2):
        df[i] = (df[i] - df[i].mean()) / (df[i].std())

#Shuffle then separate into training, validation, and test sets
x_train, x_test, y_train, y_test, m_train, m_test = train_test_split(df.iloc[:, :-7].values, df.iloc[:, -1].values,
                                                                     df.iloc[:, -2].values, test_size=0.2, shuffle = True)


sig_smp, bkg_smp = len(y_test[y_test == 1]), len(y_test[y_test == 0])
s_fac_trim = [samples[0]/sig_smp, samples[1]/bkg_smp]
print('Cut Signal: ', sig_smp*8.1330010*(10**-5)*s_fac_trim[0])
print('Cut Background: ', bkg_smp*1.4058503*(10**-4)*s_fac_trim[1])

#Data metrics
nb_samples = x_train.shape[0]
nb_features = x_train.shape[1]

#Compile and fit standard neural network
clf = define_model(nb_features)
hist_clf = clf.fit(x_train, y_train, epochs=60, batch_size=1000, validation_split=0.15)

#Make class predictions with the model
y_pred = clf.predict(x_test).flatten()

#Print various metrics and graphs based on standard neural network
report_NN(clf, hist_clf, y_pred, x_test, y_test, m_test, s_fac_trim)

#Mass Correlation Graph
plots.profile(m_test[y_test==0], y_pred[y_test==0], labels='NN Classifier on Transformed Data')
plt.show()