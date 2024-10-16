import numpy as np

# Stałe (odpowiednik 'enum' w C++)
K_UNIFORM = 0
K_GAUS = 1
K_NONE = 2

# Funkcje do formatowania wektorów na ciągi znaków
def format_vT(vT):
    return "{ " + ", ".join(str(v) for v in vT) + " }"

def format_vdouble(vd):
    return format_vT(vd)

class SVfitIntegratorMarkovChain:
    def __init__(self, init_mode: str, num_iter_burnin: int, num_iter_sampling: int, 
                 num_iter_sim_annealing_phase1: int, num_iter_sim_annealing_phase2: int, 
                 T0: float, alpha: float, num_chains: int, num_batches: int, 
                 epsilon0: float, nu: float, tree_file_name: str, verbosity: int):
        # Inicjalizacja pól
        self.integrand_ = 0
        self.x_ = 0
        self.num_integration_calls_ = 0
        self.num_moves_total_accepted_ = 0
        self.num_moves_total_rejected_ = 0
        self.error_flag_ = 0
        self.tree_file_name_ = tree_file_name
        self.tree_file_ = None
        self.tree_ = None
        
        # Obsługa initMode
        if init_mode == "uniform":
            self.init_mode_ = K_UNIFORM
        elif init_mode == "Gaus":
            self.init_mode_ = K_GAUS
        elif init_mode == "none":
            self.init_mode_ = K_NONE
        else:
            raise ValueError(f"Invalid Configuration Parameter 'init_mode' = {init_mode}, "
                             "expected to be either 'uniform', 'Gaus' or 'none' --> ABORTING !!")

        # Parametry definiujące liczbę ruchów stochastycznych na iterację
        self.num_iter_burnin = num_iter_burnin
        self.num_iter_sampling = num_iter_sampling

        ###Definicja reszty parametrów self.X = X (z odpowiednimi warunkami)###

        ###             ###            ###             ###             ###


    def __del__(self):
        # Destruktor
        if self.tree_file_ is not None:
            self.tree_file_.close()
            self.tree_file_ = None
    
    def setIntegrand(self, g, xl, xu):  ### <--- Tu wrócić, bo nie do końca rozumiem tę część kodu w cpp, ale to chyba tylko jakaś inicjalizacja
        # Przypisanie wartości do xMin_ i xMax_
        self.xMin = xl
        self.xMax = xu

        # Zmiana rozmiaru list epsilon0s_ i epsilon_
        self.epsilon0s = [self.epsilon0_] * self.numDimensions_
        self.epsilon = [0.0] * self.numDimensions_

        # Zmiana rozmiaru list p_, q_, u_, pProposal_, qProposal_
        self.p = [0.0] * (2 * self.numDimensions_)
        self.q = [0.0] * self.numDimensions_
        self.u = [0.0] * (2 * self.numDimensions_)
        self.pProposal = [0.0] * self.numDimensions_
        self.qProposal = [0.0] * self.numDimensions_

        # Zmiana rozmiaru list probSum_ i integral_
        self.probSum = [0.0] * (self.numChains_ * self.numBatches_)
        self.integral = [0.0] * (self.numChains_ * self.numBatches_)

        # Przypisanie wartości do integrand_
        self.integrand = g

    def registerCallBackFunction(self, function):
        # Dodanie funkcji do listy callBackFunctions_
        self.callBackFunctions_.append(function)

    def evalProb(self, q):
        return self.integrand(q, self.numDimensions, 0)
    
    def updateX(self, q):
        for iDimension in range(self.numDimensions):
            q_i = q[iDimension]
            self.x[iDimension] = (1 - q_i) * self.xMin[iDimension] + q_i * self.xMax[iDimension] 
    
    def initializeStartPosition_and_Momentum(self):
        #Randomly choose start position of Markov Chain in N-dimensional space
        for iDimension in range(self.numDimensions):
            isInitialized = False
            while not isInitialized:
                q0 = 0
                if self.init_mode == K_GAUS:
                    q0 = np.random.normal(0.5, 0.5)
                else:
                    q0 = np.random.uniform(0, 1)
                if q0 > 0 and q0 < 1:
                    self.q[iDimension] = q0
                    isInitialized = True
        if self.verbosity >= 2:
            print(f"<SVfitIntegratorMarkovChain::initializeStartPosition_and_Momentum>:")
            print(f"q = {format_vdouble(self.q)}")
    
    def Integrate(self, g, xl, xu, d):
        self.setIntegrand(g, xl, xu)

        if self.integrand_ is None:
            raise ValueError("<SVfitIntegratorMarkovChain>: No integrand function has been set yet --> ABORTING !!")
        
        for i in range(self.numDimensions_):
            self.xMin[i] = xl[i]
            self.xMax[i] = xu[i]
            if self.verbosity >= 1:
                print(f"dimension #{i}: min = {self.xMin[i]}, max = {self.xMax[i]}")

        ## CV: set random number generator used to initialize starting-position
        ## for each integration, in order to make integration results independent of processing history
        np.random.seed(12345)

        self.numMoves_accepted = 0
        self.numMoves_rejected = 0

        k = self.numChains * self.numBatches
        m = self.numIterSampling / self.numBatches

        numChainsrun = 0

        #Tu odpowiednik "if ( treeFileName_ != "" ) {" (zapisywanie do pliku) z biblioteką h5py (napisane przez copilota)
        #Dokładną implementację wymyślę później / naradzę się, czy korzystamy z h5py
        #Można też korzystać z roota, ale pewnie trzeba zastanowić się co będzie najszybsze
        """if self.treeFileName_ != "":
            with h5py.File(self.treeFileName_, "w") as f:
                tree_group = f.create_group("tree")
                for iDimension in range(self.numDimensions_):
                    branchName = f"x{iDimension}"
                    tree_group.create_dataset(branchName, data=self.x_[iDimension])
                tree_group.create_dataset("move", data=self.treeMove_)
                tree_group.create_dataset("integrand", data=self.treeIntegrand_)
        """


        for iChain in range(self.numChains):
            isValidStartPos = False
            if self.init_mode == K_NONE:
                self.prob = self.evalProb(q)
                if self.prob > 0:
                    isWithinBounds = True
                    for iDimension in range(self.numDimensions):
                        q_i = self.q[iDimension]
                        if not (q_i > 0 and q_i < 1):
                            isWithinBounds = False
                        if isWithinBounds:
                            isValidStartPos = True
                        else:
                            if self.verbosity >= 1:
                                print(f"<SVfitIntegratorMarkovChain>: Warning: Requested start-position = {self.format_vdouble(self.q)} not within interval ]0..1[ --> searching for valid alternative !!")
                else:
                    print(f"<SVfitIntegratorMarkovChain>: Warning: Requested start-position = {self.format_vdouble(self.q)} returned probability zero --> searching for valid alternative !!")
            iTry = 0
            while (not isValidStartPos and iTry < self.maxCallsStartingPos):
                self.initializeStartPosition_and_Momentum()
                self.prob = self.evalProb(self.q)
                if self.prob > 0:
                    isValidStartPos = True
                else:
                    if (iTry > 0 and iTry % 100000 == 0):
                        if iTry == 100000:
                            print(f"<SVfitIntegratorMarkovChain>: Warning: Requested start-position = {self.format_vdouble(self.q)} returned probability zero --> searching for valid alternative !!")
            if not isValidStartPos:
                continue

            for iMove in range(self.num_iter_burnin):
                isAccepted = False
                isValid = True
                while not isValid:
                    self.makeStochasticMove(iMove, isAccepted, isValid)

            idxBatch = iChain*self.numBatches

            for iMove in range(self.numIterSampling):
                isAccepted = False
                isValid = True
                while not isValid:
                    self.makeStochasticMove(iMove, isAccepted, isValid)

            if isAccepted:
                self.numMoves_accepted += 1
            else:
                self.numMoves_rejected += 1
            
            self.updateX(self.q)

            for callBackFunction in self.callBackFunctions:
                callBackFunction(self.x_)

            #Z h5py:
            """
            if self.tree:
                self.treeMove = iMove
                self.treeIntegrand = self.prob
                data = [self.treeMove, self.treeIntegrand]
                with h5py.File('data.h5', 'a') as f:
                    tree_group = f['tree']
                    tree_group['dataset'].resize((tree_group['dataset'].shape[0] + 1), axis=0)
                    tree_group['dataset'][-1] = data
            """

            if iMove > 0 and (iMove % m) == 0:
                idxBatch += 1
            assert idxBatch < (self.numChains * self.numBatches)
            self.probSum[idxBatch] += self.prob
    
            self.ChainsRun += 1
        
        for idxBatch in range(len(self.probSum)):
            self.integral[idxBatch] = self.probSum[idxBatch] / m
            if self.verbosity >= 1:
                print(f"integral[{idxBatch}] = {self.integral[idxBatch]}")

        #compute integral value and uncertainty
        #(eqs. (6.39) and (6.40) in [1])
        integral = 0.0
        for i in range(k):
            integral += self.integral[i]
        integral /= k

        integralErr = 0.0
        for i in range(k):
            integralErr += (self.integral[i] - integral) ** 2
        if k >= 2:
            integralErr /= (k * (k - 1))
        integralErr = np.sqrt(integralErr)
        
        if self.verbosity >= 1:
            print(f"--> returning integral = {integral} +/- {integralErr}")
        
        self.errorFlag = 0 if self.numChainsRun >= 0.5 * self.numChains else 1

        ### Tu wracamy
        self.num_integration_calls_ +=1
        self.num_moves_total_accepted_ += self.num_moves_accepted
        self.num_moves_total_rejected_ += self.num_moves_rejected

        if self.tree_:  #Zapisywanie do pliku
            self.tree_.Write()
        self.treeFile_ = None
        self.tree_ = None

        if self.verbosity >= 1:
            self.print()

    def Print(self):
        print("<SVfitIntegratorMarkovChain::print>:\n")
        for iChain in range(self.numChains):
            integral = 0
            for iBatch in range(self.numBatches):
                integral_i = self.integral[iChain * self.numBatches + iBatch]
                ##//std::cout << "batch #" << iBatch << ": integral = " << integral_i << std::endl;
                integral += integral_i
            integral /= self.numBatches
            ##//std::cout << "<integral> = " << integral << std::endl;
            integralErr = 0
            for iBatch in range(self.numBatches):
                integral_i = self.integral[iChain * self.numBatches + iBatch]
                integralErr += (integral_i - integral) ** 2
            if self.numBatches >= 2:
                integralErr /= (self.numBatches * (self.numBatches - 1))
            integralErr = np.sqrt(integralErr)
            print(f"chain #{iChain}: integral = {integral} +/- {integralErr}\n")
        print(f"moves: accepted = {self.numMoves_accepted}, rejected = {self.numMoves_rejected} (fraction = {self.numMoves_accepted / (self.numMoves_accepted + self.numMoves_rejected) * 100}%)\n")

    def sampleSphericallyRandom(self):
        uMag2 = 0
        for iDimension in range(2*self.numDimensions):
            u_i = np.random.normal(0, 1)
            self.u[iDimension] = u_i
            uMag2 += u_i ** 2
        uMag = np.sqrt(uMag2)
        for iDimension in range(2*self.numDimensions):
            self.u[iDimension] /= uMag


    ##perform "stochastic" move
    ##(eq. 24 in [2])
    def makeStochasticMove(self, idxMove, isAccepted, isValid):
        ##perform random updates of momentum components
        if idxMove < self.num_iter_sim_annealing_phase1:
            for iDimension in range(2*self.numDimensions):
                self.p[iDimension] = np.sqrt(self.T0) * np.random.normal(0, 1)
        elif idxMove < self.numIterSimAnnealingPhase1plus2:
            pMag2 = 0
            for iDimension in range(2*self.numDimensions):
                p_i = self.p[iDimension]
                pMag2 += p_i ** 2
            pMag = np.sqrt(pMag2)
            self.sampleSphericallyRandom()
            for iDimension in range(2*self.numDimensions):
                self.p[iDimension] = self.alpha * pMag * self.u[iDimension] + (1 - self.alpha2) * np.random.normal(0, 1)
        else:
            for iDimension in range(2*self.numDimensions):
                self.p[iDimension] = np.random.normal(0, 1)
        
        ##choose random step size
        exp_nu_times_C = 0
        while exp_nu_times_C <= 0 or exp_nu_times_C >= 1e+6:
            C = BreitWigner(0, 1) ### Do zaimportowania z jakiejś biblioteki
            exp_nu_times_C = np.exp(self.nu * C)
            if not (np.isnan(exp_nu_times_C) or not np.isfinite(exp_nu_times_C) or exp_nu_times_C > 1e6):
                break

        for iDimension in range(self.numDimensions):
            self.epsilon[iDimension] = self.epsilon0s[iDimension] * exp_nu_times_C
        
        ##Metropolis algorithm: move according to eq. (27) in [2]

        #update position components
        #by single step of chosen size in direction of the momentum components
        for iDimension in range(self.numDimensions):
            self.qProposal[iDimension] = self.q[iDimension] + self.epsilon[iDimension] * self.p[iDimension]

        #ensure that proposed new point is within integration region
        #(take integration region to be "cyclic")
        for iDimension in range(self.numDimensions):
            q_i = self.qProposal[iDimension]
            q_i = q_i - np.floor(q_i)
            assert q_i >= 0 and q_i <= 1
            self.qProposal[iDimension] = q_i

        #check if proposed move of Markov Chain to new position is accepted or not:
        #compute change in phase-space volume for "dummy" momentum components
        #(eqs. 25 in [2])

        probProposal = self.evalProb(self.qProposal)
        deltaE = 0
        if probProposal > 0 and self.prob > 0:
            deltaE = -np.log(probProposal / self.prob)
        elif probProposal > 0:
            deltaE = -np.inf
        elif self.prob > 0:
            deltaE = np.inf
        else:
            assert 0
        
        #Metropolis algorithm: move according to eq. (13) in [2]
        pAccept = np.exp(-deltaE)
        u = np.random.uniform(0, 1)
        if u < pAccept:
            for iDimension in range(self.numDimensions):
                self.q[iDimension] = self.qProposal[iDimension]
            self.prob = probProposal
            isAccepted = True
        else:
            isAccepted = False
