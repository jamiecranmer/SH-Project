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

#Retrieve transformed data
with open("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/transformed_data.data", 'rb') as f:
    # read the data as binary data stream
    a = pickle.load(f)
df = pd.DataFrame(a[:1000000])#895668

#Reproducability
np.random.seed(5321)

#Standardise data
for i in range(df.shape[1]-2):
        df[i] = (df[i] - df[i].mean()) / (df[i].std())

#Shuffle then separate into training, validation, and test sets
x_train, x_test, y_train, y_test, m_train, m_test = train_test_split(df.iloc[:, :-2].values, df.iloc[:, -1].values,
                                                                     df.iloc[:, -2].values, test_size=0.2, shuffle = True)


#Data metrics
nb_samples = x_train.shape[0]
nb_features = x_train.shape[1]

# Adversary factory method
def adversary_model (nb_gmm):

    # Input(s)
    i = Input(shape=(1,))
    m = Input(shape=(1,))
    
    # Hidden layer(s)
    x = Dense(4,  activation='relu')(i)
    x = Dense(8,  activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Gaussian mixture model (GMM) components
    coeffs = Dense(nb_gmm, activation='softmax') (x)  # GMM coefficients sum to one
    means  = Dense(nb_gmm, activation='sigmoid') (x)  # Means are on [0, 1]
    widths = Dense(nb_gmm, activation='softplus')(x)  # Widths are positive
    
    # Posterior p.d.f.
    pdf = layers.PosteriorLayer(nb_gmm)([coeffs, means, widths, m])

    # Build model
    return Model(inputs=[i,m], outputs=pdf, name='adversary')

# Combined factory method
def combined_model (clf, adv, lambda_reg, lr_ratio):

    # Classifier
    input_clf  = Input(shape=clf.layers[0].input_shape[1:])
    input_m    = Input(shape=(1,))
    output_clf = clf(input_clf)

    # Connect with gradient reversal layer
    gradient_reversal = layers.GradientReversalLayer(lambda_reg * lr_ratio)(output_clf)

    # Adversary
    output_adv = adv([gradient_reversal, input_m])

    # Build model
    return Model(inputs=[input_clf, input_m], outputs=[output_clf, output_adv], name='combined')

def KL (y_true, y_pred):
    """
    Kullback-Leibler loss; maximises posterior p.d.f.
    """
    return -K.log(y_pred)

#Compile and fit standard netral network
clf = define_model(nb_features)
hist_clf = clf.fit(x_train, y_train, epochs=5, batch_size=1000, validation_split=0.15)

#Make class predictions with the model
y_pred = clf.predict(x_test).flatten()

sig_smp, bkg_smp = len(y_test[y_test == 1]), len(y_test[y_test == 0])
s_fac_trim = [samples[0]/sig_smp, samples[1]/bkg_smp]
print('Cut Signal: ', sig_smp*8.1330010*(10**-5)*s_fac_trim[0])
print('Cut Background: ', bkg_smp*1.4058503*(10**-4)*s_fac_trim[1])

#Print various metrics and graphs based on standard neural network
#report_NN(clf, hist_clf, y_pred, x_test, y_test, m_test, s_fac_trim)

lam = 100                     # Regularisation parameter, lambda
nb_gmm = 6                   # Number of GMM components
loss_weights = [1.0E-08, 1.]   # Relative learning rates for classifier and adversary, resp.
lr_ratio = loss_weights[0] / loss_weights[1]
optimiser = 'adam'             # Using the Adam optimiser

# Prepare sample weights (i.e. only do mass-decorrelation for signal)
sample_weight = [np.ones(nb_samples, dtype=float), (y_train == 0).astype(float)]
sample_weight[1] *= np.sum(sample_weight[0]) / np.sum(sample_weight[1])

# Rescale mass to [0,1]
mt_train  = m_train - m_train.min()
mt_train /= mt_train.max()

# Get classifier predictions
z_train = clf.predict(x_train).flatten()
z_train_rounded = np.rint(z_train)

# Construct adversary model and fit
adv = adversary_model(nb_gmm)
adv.compile(optimiser, loss=KL)
hist_adv = adv.fit([z_train, mt_train], np.ones_like(mt_train), sample_weight=sample_weight[1], batch_size = 60, epochs=7, validation_split=0.15)

#Demonstrate learned mass ability
plots.posterior(adv, m_train[y_train == 0], z_train[y_train == 0], title='Before adversarial training')
plt.show()

# Construct combined model and fit
cmb = combined_model(clf, adv, lam, lr_ratio)
cmb.compile(optimiser, loss=['binary_crossentropy', KL], loss_weights=loss_weights)
hist_cmb = cmb.fit([x_train, mt_train], [y_train, np.ones_like(mt_train)], sample_weight=sample_weight, batch_size=500, epochs=60, validation_split=0.15)

#Predict using new classifier on train/test sets
z_train    = clf.predict(x_train).flatten()
y_pred_ANN = clf.predict(x_test).flatten()
y_pred_ANN_rounded = np.rint(y_pred_ANN)
plots.posterior(adv, m_train[y_train == 0], z_train[y_train == 0], title='After adversarial training')
plt.show()

#Mass Correlation Graph
plots.profile(m_test[y_test==0], [y_pred[y_test==0], y_pred_ANN[y_test==0]], labels=['NN classifier', 'ANN classifier'])
plt.show()

#Predictions
distribution(y_test, y_pred_ANN, xlabel="Classifier Output")
plt.show()

#Evaluate the naive model
loss, accuracy, precision, recall= clf.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
print('Loss: %.2f' % (loss))
print('Precision: %.2f' % (precision))
print('Recall: %.2f' % (recall))
print('F1: ', metrics.f1_score(y_test, y_pred_ANN_rounded))

sig_smp, bkg_smp = len(y_test[y_test == 1]), len(y_test[y_test == 0])
s_fac_trim = [samples[0]/sig_smp, samples[1]/bkg_smp]
print('Cut Signal: ', sig_smp*8.1330010*(10**-5)*s_fac_trim[0])
print('Cut Background: ', bkg_smp*1.4058503*(10**-4)*s_fac_trim[1])

#Scaling constant
SBSF = 24.838677879576565

#Sensitivity for different cuts with ANN
ss = []
t = np.linspace(0.1,0.9,20,endpoint=True)
for i in t:
    true_pos, false_pos, true_neg, false_neg = calc_Truths(y_pred_ANN, y_test, m_test, threshold = i)
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
true_pos, false_pos, true_neg, false_neg = calc_Truths(y_pred_ANN, y_test, m_test, threshold =t[ss.index(max(ss))])
eff_sig = len(true_pos)/(len(true_pos)+len(false_neg))
eff_bkg = len(false_pos)/(len(false_pos)+len(true_neg))
sensitivity = Sensitivity(true_pos, false_pos, s_fac_trim, SBSF)
print('Best effsig: ', eff_sig)
print('Best rj: ', eff_bkg)
print('Best Sensitivity: ', sensitivity)

# Mass distribution sculpting
plots.sculpting(m_test, y_test, [y_pred, y_pred_ANN], ['NN classifier', 'ANN classifier'], effsig = eff_sig)
plt.show()

plot_Sculpting(true_pos, false_pos, true_neg, false_neg)
plt.show()
fit_Exp(false_pos)

#Plot metrics
plot_ROC(y_test, y_pred_ANN)
plot_PR(y_test, y_pred_ANN)
plot_Seq_Loss(hist_cmb)