from pylorentz import Momentum4
import numpy as np

lst = []

tau_1 = Momentum4.e_eta_phi_pt(199012, -1.70762, -0.0776301, 69864)
tau_2 = Momentum4.e_eta_phi_pt(266972, -2.31905, -2.89357, 52019)
tau_3 = Momentum4.e_eta_phi_pt(139261, -0.329794, 2.39196, 52019)
tau_4 = Momentum4.e_eta_phi_pt(442340, -2.10399, -0.915685, 106287)
tau_5 = Momentum4.e_eta_phi_pt(474586, -3.10461, 0.0146134, 42473.6)
tau_6 = Momentum4.e_eta_phi_pt(42024.8, 0.50345, 1.17451, 36941.5)

parent = tau_1 + tau_2 + tau_3 + tau_4 + tau_5 + tau_6
print(tau_1.e)
print(parent.p_t)

lst.append(tau_1)
lst.append(tau_2)
lst.append(tau_3)
lst.append(tau_4)
lst.append(tau_5)
lst.append(tau_6)

parent1 = Momentum4.e_eta_phi_pt(0, 0, 0, 0)

for i in range(len(lst)):
    parent1 += lst[i]

print(parent.p_t)
print(tau_1.e)