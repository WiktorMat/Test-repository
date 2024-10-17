void SVfitIntegratorMarkovChain::print(std::ostream& stream) const
{
  stream << "<SVfitIntegratorMarkovChain::print>:" << std::endl;
  for ( unsigned iChain = 0; iChain < numChains_; ++iChain ) {
    double integral = 0.;
    for ( unsigned iBatch = 0; iBatch < numBatches_; ++iBatch ) {
      double integral_i = integral_[iChain*numBatches_ + iBatch];
      //std::cout << "batch #" << iBatch << ": integral = " << integral_i << std::endl;
      integral += integral_i;
    }
    integral /= numBatches_;
    //std::cout << "<integral> = " << integral << std::endl;

    double integralErr = 0.;
    for ( unsigned iBatch = 0; iBatch < numBatches_; ++iBatch ) {
      double integral_i = integral_[iChain*numBatches_ + iBatch];
      integralErr += square(integral_i - integral);
    }
    if ( numBatches_ >= 2 ) integralErr /= (numBatches_*(numBatches_ - 1));
    integralErr = TMath::Sqrt(integralErr);

    std::cout << " chain #" << iChain << ": integral = " << integral << " +/- " << integralErr << std::endl;
  }
  std::cout << "moves: accepted = " << numMoves_accepted_ << ", rejected = " << numMoves_rejected_
            << " (fraction = " << (double)numMoves_accepted_/(numMoves_accepted_ + numMoves_rejected_)*100.
            << "%)" << std::endl;
}