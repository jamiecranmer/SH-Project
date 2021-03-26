import numpy as np

class Particle():

    def __init__(self, pt, eta, phi, energy):

        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.energy = energy

class Jet():

    def __init__(self, pt, eta, phi, energy, DLR1):

        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.energy = energy
        self.DLR1 = DLR1