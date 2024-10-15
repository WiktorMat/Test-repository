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
        
        ### Jeszcze do przepisania:
        
        for ( unsigned idxBatch = 0; idxBatch < probSum_.size(); ++idxBatch ) {
    integral_[idxBatch] = probSum_[idxBatch]/m;
    if ( verbosity_ >= 1 ) std::cout << "integral[" << idxBatch << "] = " << integral_[idxBatch] << std::endl;
  }

//--- compute integral value and uncertainty
//   (eqs. (6.39) and (6.40) in [1])
  integral = 0.;
  for ( unsigned i = 0; i < k; ++i ) {
    integral += integral_[i];
  }
  integral /= k;

  integralErr = 0.;
  for ( unsigned i = 0; i < k; ++i ) {
    integralErr += square(integral_[i] - integral);
  }
  if ( k >= 2 ) integralErr /= (k*(k - 1));
  integralErr = TMath::Sqrt(integralErr);

  if ( verbosity_ >= 1 ) std::cout << "--> returning integral = " << integral << " +/- " << integralErr << std::endl;

  errorFlag_ = ( numChainsRun_ >= 0.5*numChains_ ) ? 0 : 1;

  ++numIntegrationCalls_;
  numMovesTotal_accepted_ += numMoves_accepted_;
  numMovesTotal_rejected_ += numMoves_rejected_;

  if ( tree_ ) {
    tree_->Write();
  }
  delete treeFile_;
  treeFile_ = 0;
  //delete tree_;
  tree_ = 0;

  if ( verbosity_ >= 1 ) print(std::cout);
}


        
        ###




        #Placeholder dla właściwego kodu
        integral = 0.0
        integralErr = 0.0
        return integral, integralErr
