import numpy as np

class MeasuredTauLepton:
    def __init__(self, fullsetup = False, type = None, pt = 0., eta = 0., phi = 0., mass = 0., decayMode = -1):
        if type is None:
            self.type = MeasuredTauLepton.kUndefinedDecayType
        else:
            self.type = type
        self.pt_ = pt
        self.eta_ = eta
        self.phi_ = phi
        self.mass_ = mass
        self.decayMode_ = decayMode
        if fullsetup:
            return
            #Zostawiam miejsce na pełen setup z drugiego konstruktora
            #Na tyle prosty, że przepiszę po potwierdzeniu
        
        



        #stałe i inicjalizacja z domyślnego konstruktora


        return

    
    #Tu wznowimy następnym razem:)