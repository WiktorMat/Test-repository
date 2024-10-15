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