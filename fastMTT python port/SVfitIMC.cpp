    void SVfitIntegratorMarkovChain::setIntegrand(gPtr_C g, const double* xl, const double* xu, unsigned d)
{
  numDimensions_ = d;

  delete [] x_;
  x_ = new double[numDimensions_];

  xMin_.resize(numDimensions_);
  xMax_.resize(numDimensions_);
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    xMin_[iDimension] = xl[iDimension];
    xMax_[iDimension] = xu[iDimension];
  }

  epsilon0s_.resize(numDimensions_);
  epsilon_.resize(numDimensions_);
  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    epsilon0s_[iDimension] = epsilon0_;
  }

  p_.resize(2*numDimensions_);   // first N entries = "significant" components, last N entries = "dummy" components
  q_.resize(numDimensions_);     // "potential energy" E(q) depends in the first N "significant" components only
  prob_ = 0.;

  u_.resize(2*numDimensions_);   // first N entries = "significant" components, last N entries = "dummy" components
  pProposal_.resize(numDimensions_);
  qProposal_.resize(numDimensions_);

  probSum_.resize(numChains_*numBatches_);
  for ( vdouble::iterator probSum_i = probSum_.begin();
  probSum_i != probSum_.end(); ++probSum_i ) {
    (*probSum_i) = 0.;
  }
  integral_.resize(numChains_*numBatches_);

  integrand_ = g;
}

void SVfitIntegratorMarkovChain::integrate(gPtr_C g, const double* xl, const double* xu, unsigned d, double& integral, double& integralErr)
{
  setIntegrand(g, xl, xu, d);

  if ( !integrand_ ) {
    std::cerr << "<SVfitIntegratorMarkovChain>:"
              << "No integrand function has been set yet --> ABORTING !!\n";
    assert(0);
  }

  for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
    xMin_[iDimension] = xl[iDimension];
    xMax_[iDimension] = xu[iDimension];
    if ( verbosity_ >= 1 ) {
      std::cout << "dimension #" << iDimension << ": min = " << xMin_[iDimension] << ", max = " << xMax_[iDimension] << std::endl;
    }
  }

//--- CV: set random number generator used to initialize starting-position
//        for each integration, in order to make integration results independent of processing history
  rnd_.SetSeed(12345);

  numMoves_accepted_ = 0;
  numMoves_rejected_ = 0;

  unsigned k = numChains_*numBatches_;
  unsigned m = numIterSampling_/numBatches_;

  numChainsRun_ = 0;

  if ( treeFileName_ != "" ) {
    treeFile_ = new TFile(treeFileName_.data(), "RECREATE");
    tree_ = new TTree("tree", "Markov Chain transitions");
    for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
      std::string branchName = Form("x%u", iDimension);
      tree_->Branch(branchName.data(), &x_[iDimension]);
    }
    tree_->Branch("move", &treeMove_);
    tree_->Branch("integrand", &treeIntegrand_);
  }

  for ( unsigned iChain = 0; iChain < numChains_; ++iChain ) {
    bool isValidStartPos = false;
    if ( initMode_ == kNone ) {
      prob_ = evalProb(q_);
      if ( prob_ > 0. ) {
      bool isWithinBounds = true;
      for ( unsigned iDimension = 0; iDimension < numDimensions_; ++iDimension ) {
        double q_i = q_[iDimension];
        if ( !(q_i > 0. && q_i < 1.) ) isWithinBounds = false;
      }
      if ( isWithinBounds ) {
        isValidStartPos = true;
      } else {
        if ( verbosity_ >= 1 ) {
          std::cerr << "<SVfitIntegratorMarkovChain>:"
                    << "Warning: Requested start-position = " << format_vdouble(q_) << " not within interval ]0..1[ --> searching for valid alternative !!\n";
        }
      }
      } else {
        if ( verbosity_ >= 1 ) {
          std::cerr << "<SVfitIntegratorMarkovChain>:"
                    << "Warning: Requested start-position = " << format_vdouble(q_) << " returned probability zero --> searching for valid alternative !!";
        }
      }
    }
    unsigned iTry = 0;
    while ( !isValidStartPos && iTry < maxCallsStartingPos_ ) {
      initializeStartPosition_and_Momentum();
      prob_ = evalProb(q_);
      if ( prob_ > 0. ) {
        isValidStartPos = true;
      } else {
        if ( iTry > 0 && (iTry % 100000) == 0 ) {
          if ( iTry == 100000 ) std::cout << "<SVfitIntegratorMarkovChain::integrate>:" << std::endl;
          std::cout << "try #" << iTry << ": did not find valid start-position yet." << std::endl;
        }
      }
      ++iTry;
    }
    if ( !isValidStartPos ) continue;

    for ( unsigned iMove = 0; iMove < numIterBurnin_; ++iMove ) {
//--- propose Markov Chain transition to new, randomly chosen, point
      bool isAccepted = false;
      bool isValid = true;
      do {
  makeStochasticMove(iMove, isAccepted, isValid);
      } while ( !isValid );
    }

    unsigned idxBatch = iChain*numBatches_;

    for ( unsigned iMove = 0; iMove < numIterSampling_; ++iMove ) {
//--- propose Markov Chain transition to new, randomly chosen, point;
//    evaluate "call-back" functions at this point
      bool isAccepted = false;
      bool isValid = true;
      do {
        makeStochasticMove(numIterBurnin_ + iMove, isAccepted, isValid);
      } while ( !isValid );
      if ( isAccepted ) {
        ++numMoves_accepted_;
      } else {
        ++numMoves_rejected_;
      }

      updateX(q_);
      for ( std::vector<const ROOT::Math::Functor*>::const_iterator callBackFunction = callBackFunctions_.begin();
            callBackFunction != callBackFunctions_.end(); ++callBackFunction ) {
        (**callBackFunction)(x_);
      }

      if ( tree_ ) {
        treeMove_ = iMove;
        treeIntegrand_ = prob_;
        tree_->Fill();
      }

      if ( iMove > 0 && (iMove % m) == 0 ) ++idxBatch;
      assert(idxBatch < (numChains_*numBatches_));
      probSum_[idxBatch] += prob_;
    }

    ++numChainsRun_;
  }

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