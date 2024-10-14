import numpy as np

class SVfitIntegratorMarkovChain:
    def __init__(self) -> None:
        self.numIterBurnin = 1000
        self.numIterSampling = 10000
        self.numIterSimAnnealingPhase1 = 2500
        self.numIterSimAnnealingPhase2 = 2500
        self.T0 = 15.
        self.alpha = 0.995
        self.numChains = 1
        self.numBatches = 1
        self.epsilon0 = 1.e-2
        self.nu = 0.71
        
        pass