from Plot_Quantities import *
import pickle
import pandas as pd
import numpy as np

#Retrieve transformed data
with open("C:/Users/jamie/OneDrive/Documents/Python Scripts/SHproject2/transformed_data.data", 'rb') as f:
    # read the data as binary data stream
    a = pickle.load(f)
df = pd.DataFrame(a)

df.columns = ['g1Pt','g1Eta','g1Phi','g1Energy','g2Pt','g2Eta','g2Phi','g2Energy','J1Pt','J1Eta','J1Phi','J1Energy','J1DLR1',
              'J12Pt','J2Eta','J2Phi','J2Energy','J2DLR1','J3Pt','J3Eta','J3Phi','J3Energy','J3DLR1','J4Pt','J4Eta','J4Phi','J4Energy','J4DLR1',
              'Parent Pt','Parent Eta','Parent Energy','Absolute Angle','Mass','targs']

#Isolate interesting variables
ys = df['targs'].values
ms = df['Mass'].values
ags = df['Absolute Angle'].values

#Isolate leading photon and data
df_g1 = df[['g1Pt','g1Eta','g1Phi','g1Energy']]
pt1 = df_g1['g1Pt'].values
eta1 = df_g1['g1Eta'].values
phi1 = df_g1['g1Phi'].values
energy1 = df_g1['g1Energy'].values

plt.rcParams.update({'font.size': 20})

#Plot LP quantities
plot_Feat_Hist(pt1[ys == 1], pt1[ys == 0], 'pt', 'Leading Photon', (0,250))
plot_Feat_Hist(eta1[ys == 1], eta1[ys == 0], 'eta', 'Leading Photon', (-2.8,2.8))
plot_Feat_Hist(phi1[ys == 1], phi1[ys == 0], 'phi', 'Leading Photon',(-3.5,3.5))
plot_Feat_Hist(energy1[ys == 1], energy1[ys == 0], 'energy', 'Leading Photon',(0,400))

#Isolate secondary photon and data
df_g2 = df[['g2Pt','g2Eta','g2Phi','g2Energy']]
pt2 = df_g2['g2Pt'].values
eta2 = df_g2['g2Eta'].values
phi2 = df_g2['g2Phi'].values
energy2 = df_g2['g2Energy'].values

#Plot SP quantities
plot_Feat_Hist(pt2[ys == 1], pt2[ys == 0], 'pt', 'Secondary Photon',(0,125))
plot_Feat_Hist(eta2[ys == 1], eta2[ys == 0], 'eta', 'Secondary Photon',(-2.8,2.8))
plot_Feat_Hist(phi2[ys == 1], phi2[ys == 0], 'phi', 'Secondary Photon',(-3.5,3.5))
plot_Feat_Hist(energy2[ys == 1], energy2[ys == 0], 'energy', 'Secondary Photon',(0,250))

#Isolate parent properties
df_P = df[['Parent Pt','Parent Eta','Parent Energy']]
Ppt = df_P['Parent Pt'].values
Peta = df_P['Parent Eta'].values
PE = df_P['Parent Energy'].values

#Plot parent properties
plot_Feat_Hist(Ppt[ys == 1], Ppt[ys == 0], 'pt', 'Parent Particle',(0,300))
plot_Feat_Hist(Peta[ys == 1], Peta[ys == 0], 'eta', 'Parent Particle',(-6,6))
plot_Feat_Hist(PE[ys == 1], PE[ys == 0], 'energy', 'Parent Particle',(0,500))
plot_Feat_Hist(ms[ys == 1], ms[ys == 0], 'mass', 'Parent Particle',(90,180))

#prepare heatmap datasets
plt.clf()
plt.rcParams.update({'font.size': 15})
bkg_df = df[df['targs'] == 0] 
photon_corrs = bkg_df[['g1Pt','g1Eta','g1Phi','g1Energy','g2Pt','g2Eta','g2Phi','g2Energy', 'Mass']]
jet_corrs = bkg_df[['J1Pt','J1Eta','J1Phi','J1Energy','J1DLR1',
              'J12Pt','J2Eta','J2Phi','J2Energy','J2DLR1','J3Pt','J3Eta','J3Phi','J3Energy','J3DLR1','J4Pt','J4Eta','J4Phi','J4Energy','J4DLR1', 'Mass']]
parent_corrs = bkg_df[['Parent Pt','Parent Eta','Parent Energy','Absolute Angle','Mass']]
heatmap = sns.heatmap(photon_corrs.corr(), xticklabels=True, yticklabels=True, vmin=0, vmax=1, annot = True, fmt = '.2f')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=20) 
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=20) 
plt.show()
heatmap = sns.heatmap(jet_corrs.corr(), xticklabels=True, yticklabels=True, vmin=0, vmax=1, annot = True, fmt = '.2f')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=20) 
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=20) 
plt.show()
heatmap = sns.heatmap(parent_corrs.corr(), xticklabels=True, yticklabels=True, vmin=0, vmax=1, annot = True, fmt = '.2f')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=20) 
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=20) 
plt.show()

#For 2D histograms
plt.figure(figsize=(6,6), dpi= 100)
plt.rcParams.update({'font.size': 25})

xlims, ylims = [80, 150], [35, 140]
plt.hist2d(bkg_df['Mass'], bkg_df['g1Pt'], normed = True, bins = 150, range = [xlims,ylims])
plt.ylabel('Leading Photon Transverse Momentum [GeV]')
plt.xlabel('Mass [GeV]')
plt.show()

xlims, ylims = [80, 150], [35, 140]
plt.hist2d(bkg_df['Mass'], bkg_df['g1Energy'], normed = True, bins = 150, range = [xlims,ylims])
plt.ylabel('Leading Photon Energy [GeV]')
plt.xlabel('Mass [GeV]')
plt.show()

xlims, ylims = [80, 180], [0, 6]
plt.hist2d(bkg_df['Mass'], bkg_df['Absolute Angle'], normed = True, bins = 150,  range = [xlims,ylims])
plt.ylabel('Photon Angular Separation {}'.format(r'$\Delta \theta_{\gamma\gamma}$'))
plt.xlabel('Mass [GeV]')
plt.show()

xlims, ylims = [80, 180], [80, 250]
plt.hist2d(bkg_df['Mass'], bkg_df['Parent Energy'], normed = True, bins = 150, range = [xlims,ylims])
plt.ylabel('Parent Particle Energy [GeV]')
plt.xlabel('Mass [GeV]')
plt.show()