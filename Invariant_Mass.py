import csv
import matplotlib.pyplot as plt
from pylorentz import Momentum4
import numpy as np
import sklearn.metrics as metrics
from scipy import stats 
import pickle


def load_Data():

    #Read in signal data
    f = open('C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/ttH.csv', 'r')
    sig_data = list(csv.reader(f,quoting=csv.QUOTE_NONNUMERIC))
    f.close()

    #Read in background data similarly
    f1 = open('C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/bkg_yy_tt.csv')
    bkg_data = list(csv.reader(f1,quoting=csv.QUOTE_NONNUMERIC))
    f1.close()
    
    #Put all gamma1 gamma2 data into numpy array for easy neural network input
    #Dimensions N by 28 as the minimum number of jets we want is 4 (so 8 + 4*5)
    sig_gjs = np.zeros((len(sig_data), 28))
    bkg_gjs = np.zeros((len(bkg_data), 28))

    for i in range(len(sig_data)):
        
        if len(sig_data[i]) >= 28:
            sig_gjs[i] = sig_data[i][:28]

    for i in range(len(bkg_data)):

        if len(bkg_data[i]) >= 28:
            bkg_gjs[i] = bkg_data[i][:28]
    
    return sig_gjs, bkg_gjs

def write_All(sig_gjs, bkg_gjs):

    #Write all data to file
    sig_gjs, bkg_gjs, num_sig, num_bkg = load_Data()

    all_gjs = np.concatenate((sig_gjs, bkg_gjs))

    with open(("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/all_data.data"), "wb") as f:
        pickle.dump(all_gjs, f)

def generate_Features():

    #Use this function to set up data correctly

    #Retrieve data
    with open("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/all_data.data", 'rb') as f:
        # read the data as binary data stream
        a = pickle.load(f)

    df = pd.DataFrame(a)

    #For GeV
    df[0] = df[0]/1000   #g1pt
    df[3] = df[3]/1000   #g1E
    df[4] = df[4]/1000   #g2pt
    df[7] = df[7]/1000   #g2E
    df[8] = df[8]/1000   #J1pt
    df[11] = df[11]/1000 #J1E
    df[13] = df[13]/1000   #J2pt
    df[16] = df[16]/1000   #J2E
    df[18] = df[18]/1000   #J3pt
    df[21] = df[21]/1000   #J3E
    df[23] = df[23]/1000   #J4pt
    df[26] = df[26]/1000   #J4E

    #Extract the parent quantities    
    m, pt, energy, eta, phi = parent_Quantities(df.values[:, :8])
    df['p_pt'] = pt
    df['p_eta'] = eta
    df['p_e'] = energy

    #Absolute angle feature
    del_r = np.zeros((df.shape[0]))
    for i in range(df.shape[0]):
        del_r[i] = np.sqrt((df.values[i, 1]-df.values[i,5])**2 + (df.values[i, 2]-df.values[i,6])**2)
    df['ab_angle'] = del_r

    #Add labels and masses
    correct = np.ones(len(sig_gjs))
    incorrect = np.zeros(len(bkg_gjs))
    targets = np.concatenate((correct, incorrect))
    df['inv_mass'] = m
    df['targets'] = targets

    #Trim invalid rows (where mass is 0)
    idx = np.array([])
    for i in range(df.shape[0]):
        if m[i] ==0:
            idx = np.append(idx,i)
        elif m[i] != 0:
            pass
    df.drop(idx.astype(int), inplace = True)

    #Add transformed data to file
    with open(("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/transformed_data.data"), "wb") as f:
        pickle.dump(df.values, f)

def mass_Scaled():

    #use this function to perform mass scaling

    #Retrieve transformed data
    with open("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/transformed_data.data", 'rb') as f:
        # read the data as binary data stream
        a = pickle.load(f)
    df = pd.DataFrame(a)

    #For mass scaling (assumes mass entry is second last column)
    for i in range(df.shape[0]):
        df.iloc[i,0] = df.iloc[i,0]/df.iloc[i, -2]
        df.iloc[i,3] = df.iloc[i,3]/df.iloc[i, -2]
        df.iloc[i,4] = df.iloc[i,4]/df.iloc[i, -2]
        df.iloc[i,7] = df.iloc[i,7]/df.iloc[i, -2]
        df.iloc[i,8] = df.iloc[i,8]/df.iloc[i, -2]
        df.iloc[i,11] = df.iloc[i,11]/df.iloc[i, -2]
        df.iloc[i,13] = df.iloc[i,13]/df.iloc[i, -2]
        df.iloc[i,16] = df.iloc[i,16]/df.iloc[i, -2]
        df.iloc[i,18] = df.iloc[i,18]/df.iloc[i, -2]
        df.iloc[i,21] = df.iloc[i,21]/df.iloc[i, -2]
        df.iloc[i,23] = df.iloc[i,23]/df.iloc[i, -2]
        df.iloc[i,26] = df.iloc[i,26]/df.iloc[i, -2]
    

    #Add mass scaled features data to file
    with open(("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/mass_scaled_data.data"), "wb") as f:
        pickle.dump(df.values, f)


def lorentzify(lst):

    #Used to turn a list into PyLorentz quantities

    gamma_objects = []

    #Separate gamma columns
    each_gamma = np.split(lst, 2)

    #Change each gamma into a PyLorentz object
    for j in range(2):
        gamma_objects.append(Momentum4.e_eta_phi_pt(each_gamma[j][3],each_gamma[j][1], each_gamma[j][2], each_gamma[j][0]))

    return gamma_objects


def parent_Quantities(lst):

    #Use PyLorentz to calculate parent particle quantities

    #Set memory placeholders for each list to avoid appends
    inv_masses = np.zeros(len(lst))
    trans_momenta = np.zeros(len(lst))
    energies = np.zeros(len(lst))
    etas = np.zeros(len(lst))
    phis = np.zeros(len(lst))

    for i in range(len(lst)):

        #Turn list into PyLorentz objects
        gammas = lorentzify(lst[i])
        parent = gammas[0] + gammas[1]

        #Calculate quantities
        inv_masses[i] = parent.m
        trans_momenta[i] = parent.p_t
        energies[i] = parent.e
        etas[i] = parent.eta
        phis[i] = parent.phi
    
    return inv_masses, trans_momenta, energies, etas, phis

def ind_Quantities(lst):

    #separates input into its individual quantities

    trans_momenta = np.zeros(len(lst))
    energies = np.zeros(len(lst))
    etas = np.zeros(len(lst))
    phis = np.zeros(len(lst))

    for i in range(len(lst)):

        trans_momenta[i] = lst[i][0]
        etas[i] = lst[i][1]
        phis[i] = lst[i][2]
        energies[i] = lst[i][3]

    return trans_momenta, energies, etas, phis