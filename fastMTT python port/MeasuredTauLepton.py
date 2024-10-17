import numpy as np

class MeasuredTauLepton:
    def __init__(self, fullsetup = False, type = None, pt = 0., eta = 0., phi = 0., mass = 0., decayMode = -1):
        if type is None:
            self.type = MeasuredTauLepton.kUndefinedDecayType
        else:
            self.type = type
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass
        self.decayMode = decayMode
        if fullsetup:
            return
            #Zostawiam miejsce na pełen setup z drugiego konstruktora
            #Na tyle prosty, że przepiszę po potwierdzeniu
        
        



        #stałe i inicjalizacja z domyślnego konstruktora


        return

    def __copy__(self):
        return
        #Tu zrobimy konstruktor kopiujący

    ###Destruktor

    ##Dalej kod jest raczej prosty i głównie inicjalizuje, więc myślę, że to po prostu przepiszemy jeden do jednego