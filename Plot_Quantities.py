from Invariant_Mass import *
from plots import *
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.optimize
from atlasify import atlasify
from scipy.optimize import curve_fit


def plot_Feat_Hist(sig_quant, bkg_quant, quantity, folder, xlims):

    if quantity == 'mass':
        plt.xlabel('{} Mass [GeV]'.format(folder))
        path = ('C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/{}/Inv_Masses.png').format(folder)

    elif quantity == 'pt':
        plt.xlabel('{} Transverse Momentum [GeV]'.format(folder))
        path = ('C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/{}/Momenta.png').format(folder)

    elif quantity == 'energy':
        plt.xlabel('{} Energy [GeV]'.format(folder))
        path = ('C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/{}/Energies.png').format(folder)

    elif quantity == 'eta':
        plt.xlabel('{} Rapidity [{}]'.format(folder, r'$\eta$'))
        path = ('C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/{}/Etas.png').format(folder)

    elif quantity == 'phi':
        plt.xlabel('{} Azimuthal Angle [{}]'.format(folder, r'$\phi$'))
        path = ('C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/{}/Phis.png').format(folder)

    bins = np.linspace(xlims[0],xlims[1],100,endpoint=True)
    plt.ylabel('Fraction of Events')

    plt.hist(bkg_quant, color = 'darkblue', bins = bins, density = True, histtype='step', linewidth = 2, label = 'Background')
    plt.hist(sig_quant, color = 'red', bins = bins, density = True, histtype='step', linewidth = 2, label = 'Signal')

    plt.legend()
    atlasify('Work in Progress', enlarge=1.1)
    plt.savefig(path)
    plt.show()


def plot_ROC(targets, preds):

    plt.rcParams.update({'font.size': 20})
    fpr, tpr, _ = metrics.roc_curve(targets, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right',prop={'size': 15})
    plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim((0,1))
    # plt.ylim((0,1))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_PR(targets, preds):

    p, r, _ = metrics.precision_recall_curve(targets, preds)
    plt.rcParams.update({'font.size': 20})
    plt.plot(r, p)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    #plt.ylim((0,1))
    plt.show()

def plot_Accuracy(model):

    plt.rcParams.update({'font.size': 20})
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim((0,1))
    plt.legend(['train', 'validation'])
    plt.show()

def plot_Precision(model):

    plt.rcParams.update({'font.size': 20})
    plt.plot(model.history['precision'], label = 'train')
    plt.plot(model.history['val_precision'], label = 'validation')
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    #plt.ylim((0,1))
    plt.legend()
    plt.show()

def plot_Recall(model):

    plt.rcParams.update({'font.size': 20})
    plt.plot(model.history['recall'], label = 'train')
    plt.plot(model.history['val_recall'], label = 'validation')
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    #plt.ylim((0,1))
    plt.legend()
    plt.show()

def plot_Loss(model):

    plt.rcParams.update({'font.size': 20})
    plt.plot(model.history['loss'], label = 'Training Set')
    plt.plot(model.history['val_loss'], label = 'Validation Set')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_Seq_Loss(model):
    
    plt.rcParams.update({'font.size': 20})
    plt.plot(model.history['sequential_1_loss'], label = 'Training Set')
    plt.plot(model.history['val_sequential_1_loss'], label = 'Validation Set')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def Sensitivity(true_pos, false_pos, s_fac_trim, SBSF):

    #Normalisation scaling data
    SB_data = 5907

    s, b, band_bkg = 0, 0, 0
    for i in range(len(true_pos)):
        if (true_pos[i] > 121) and (true_pos[i] < 129):
            s += 1
    for i in range(len(false_pos)):
        if (false_pos[i] > 121) and (false_pos[i] < 129):
            b += 1
        elif (false_pos[i] >= 90 and false_pos[i] <= 121) or (false_pos[i] >= 129 and false_pos[i] <= 170):
            band_bkg += 1

    #For correct background normalisation
    band_bkg = band_bkg*(1.4058503*(10**-4))*s_fac_trim[1]

    s = s*s_fac_trim[0]*(8.1330010*(10**-5))
    b = b*s_fac_trim[1]*(1.4058503*(10**-4))*SBSF
    n = (s+b)
    sensitivity = np.sqrt(2*(n*np.log(n/b)+b-n))

    return sensitivity

def calc_Truths(y_pred, y_test, m, threshold):

    #Calculates truch vales upon a threshold

    true_pos = np.array([])
    false_pos = np.array([])
    true_neg = np.array([])
    false_neg = np.array([])

    for i in range(len(y_test)):

        if y_test[i] == 1 and y_pred[i] >= threshold:
            true_pos = np.append(true_pos, m[i])
        elif y_test[i] == 0 and y_pred[i] >= threshold:
            false_pos = np.append(false_pos, m[i])
        elif y_test[i] == 1 and y_pred[i] < threshold:
            false_neg = np.append(false_neg, m[i])
        elif y_test[i] == 0 and y_pred[i] < threshold:
            true_neg = np.append(true_neg, m[i])

    return true_pos, false_pos, true_neg, false_neg

def plot_Sculpting(true_pos, false_pos, true_neg, false_neg):

    plt.rcParams.update({'font.size': 20})
            
    plt.hist(true_pos, bins = np.linspace(100, 150, 25, endpoint=True), density = True, color = 'r', alpha = 0.5, label = 'True Positives (Real Signals)')
    plt.hist(false_pos, bins = np.linspace(100, 150, 25, endpoint=True), density = True, color = 'b', alpha = 0.5, label = 'False Positives (Background Classed as Signal)')
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Fraction of Events / 2 GeV')
    plt.xlim((100, 150))
    plt.legend(prop={'size': 15})
    atlasify('Work in Progress',enlarge=1.1)
    plt.show()


def neg_exponential(x, a, b):
    return a*np.exp(-b*x)

def fit_Exp(false_pos):

    #Plot false positives and fit a negative exponential to them

    #Histograms of signal and background and combined
    xs = np.linspace(110, 140, 15, endpoint=True)
    l = plt.hist(false_pos, density = True, bins = xs, alpha = 0.5, label = 'Background Data Passing Cut')

    #Fit exponential decay to the background histogram
    x1, y1 = l[1], l[0]
    y1 = np.append(y1, y1[-1])
    pars, cov = curve_fit(f=neg_exponential, xdata=x1, ydata=y1, p0=[0, 0], bounds=(-np.inf, np.inf))
    n_exp = neg_exponential(x1, *pars)

    #Plot Fit
    plt.plot(xs, n_exp, linestyle = '--', color='blue', label = 'Background Fit')
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Fraction of Events / 2 GeV')
    atlasify('Work in Progress',enlarge=1.1)
    plt.legend()
    plt.show()


#For DSCB fit (Not in full project)
# def DSCB(x, a_l, a_h, n_l, n_h, mu, sigma, N):

#     t = (x - mu)/sigma
#     A = a_l/n_l
#     B = (n_l/a_l) - a_l - t
#     C = a_h/n_h
#     D = (n_h/a_h) - a_h + t

#     if (-a_l <= t) and (a_h >= t):
#         sol = np.exp(-0.5*(t**2))
#     elif (t < -a_l):
#         sol = (np.exp(-0.5*(a_l**2))) * (A*B)**(-n_l)
#     elif (t > a_h):
#         sol = (np.exp(-0.5*(a_h**2))) * (C*D)**(-n_h)

#     return N*sol

# def total_Fit(true_pos, true_neg):

#     #Histograms of signal and background and combined
#     xs = np.linspace(110, 140, 60, endpoint=True)
#     lsig = plt.hist(true_pos, density = True, bins = xs, alpha = 0.5, label = 'Signal Data Only')
#     lbkg = plt.hist(true_neg, density = True, bins = xs, alpha = 0.5, label = 'Background Data Only')
#     #plt.clf()

#     #Fit exponential decay to the background histogram
#     x1, y1 = lbkg[1], lbkg[0]
#     y1 = np.append(y1, y1[-1])
#     pars, cov = curve_fit(f=neg_exponential, xdata=x1, ydata=y1, p0=[0, 0], bounds=(-np.inf, np.inf))
#     n_exp = neg_exponential(x1, *pars)

#     #Use guesses of parameters for crystal ball curve
#     x2, y2 = lsig[1], lsig[0]
#     y2 = np.append(y2, y2[-1])
#     dscb = np.array([])
#     for i in x2:
#         p = DSCB(i, 1.79, 1.59, 4.61, 17.28, 125.15, 1.46, np.amax(y2))
#         dscb = np.append(dscb, p)

#     #Scaling for total fit
#     tot_fit = np.array([])
#     for i in range(len(dscb)):
#         s = dscb[i]*(1.4058503*(10**-4))
#         b = n_exp[i]*(8.1330010*(10**-5))
#         weight =  np.log(1+(s/np.sqrt(b)))
#         tot_fit = np.append(tot_fit, weight*(dscb[i]+n_exp[i]))

#     #Plot fits
#     plt.plot(xs, n_exp, linestyle = '--', color='blue', label = 'Background Fit')
#     plt.plot(xs, dscb, linestyle = '--', color ='green', label = 'Signal Fit')
#     plt.plot(xs, tot_fit+n_exp, color = 'red', label = 'Signal + Background Fits')
#     plt.xlabel('Mass [GeV]')
#     plt.ylabel('Fraction of Events / 0.5 GeV')
#     atlasify('Work in Progress', 'Ln(1+S/B) weighted sum')
#     plt.legend()
#     plt.show()
########################################