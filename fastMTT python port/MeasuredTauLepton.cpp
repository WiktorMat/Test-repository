#include "TauAnalysis/ClassicSVfit/interface/MeasuredTauLepton.h"
#include "TauAnalysis/ClassicSVfit/interface/svFitAuxFunctions.h"

#include <TMath.h>

using namespace classic_svFit;

MeasuredTauLepton::MeasuredTauLepton()
  : type_(kUndefinedDecayType),
    pt_(0.),
    eta_(0.),
    phi_(0.),
    mass_(0.),
    decayMode_(-1)
{
  initialize();
}

MeasuredTauLepton::MeasuredTauLepton(int type, double pt, double eta, double phi, double mass, int decayMode)
  : type_(type),
    pt_(pt),
    eta_(eta),
    phi_(phi),
    mass_(mass),
    decayMode_(decayMode)
{
  //std::cout << "<MeasuredTauLepton>:" << std::endl;
  //std::cout << " Pt = " << pt_ << ", eta = " << eta_ << ", phi = " << phi_ << ", mass = " << mass_ << std::endl;
  double minVisMass = electronMass;
  double maxVisMass = tauLeptonMass;
  std::string type_string;
  if ( type_ == kTauToElecDecay ) {
    minVisMass = electronMass;
    maxVisMass = minVisMass;
  } else if ( type_ == kTauToMuDecay ) {
    minVisMass = muonMass;
    maxVisMass = minVisMass;
  } else if ( type_ == kTauToHadDecay ) {
    if ( decayMode_ == -1 ) {
      minVisMass = chargedPionMass;
      maxVisMass = 1.5;
    } else if ( decayMode_ == 0 ) {
      minVisMass = chargedPionMass;
      maxVisMass = minVisMass;
    } else {
      minVisMass = 0.3;
      maxVisMass = 1.5;
    }
  }
  preciseVisMass_ = mass_;
  if ( preciseVisMass_ < (0.9*minVisMass) || preciseVisMass_ > (1.1*maxVisMass) ) {
    std::string type_string;
    if      ( type_ == kTauToElecDecay ) type_string = "tau -> electron decay";
    else if ( type_ == kTauToMuDecay   ) type_string = "tau -> muon decay";
    else if ( type_ == kTauToHadDecay  ) type_string = "tau -> had decay";
    else {
      std::cerr << "Error: Invalid type " << type_ << " declared for leg: Pt = " << pt_ << ", eta = " << eta_ << ", phi = " << phi_ << ", mass = " << mass_ << " !!" << std::endl;
      assert(0);
    }
    std::cerr << "Warning: " << type_string << " declared for leg: Pt = " << pt_ << ", eta = " << eta_ << ", phi = " << phi_ << ", mass = " << mass_ << " !!" << std::endl;
    std::cerr << " (mass expected in the range = " << minVisMass << ".." << maxVisMass << ")" << std::endl;
  }
  if ( preciseVisMass_ < minVisMass ) preciseVisMass_ = minVisMass;
  if ( preciseVisMass_ > maxVisMass ) preciseVisMass_ = maxVisMass;
  initialize();
  //std::cout << " En = " << energy_ << ", Px = " << px_ << ", Py = " << py_ << ", Pz = " << pz_ << std::endl;
}

MeasuredTauLepton::MeasuredTauLepton(const MeasuredTauLepton& measuredTauLepton)
  : type_(measuredTauLepton.type()),
    pt_(measuredTauLepton.pt()),
    eta_(measuredTauLepton.eta()),
    phi_(measuredTauLepton.phi()),
    mass_(measuredTauLepton.mass()),
    decayMode_(measuredTauLepton.decayMode())
{
  preciseVisMass_ = measuredTauLepton.mass();

  initialize();
}

MeasuredTauLepton::~MeasuredTauLepton()
{
}

int MeasuredTauLepton::type() const { return type_; }

double MeasuredTauLepton::pt() const { return pt_; }
double MeasuredTauLepton::eta() const { return eta_; }
double MeasuredTauLepton::phi() const { return phi_; }
double MeasuredTauLepton::mass() const { return preciseVisMass_; }

double MeasuredTauLepton::energy() const { return energy_; }
double MeasuredTauLepton::px() const { return px_; }
double MeasuredTauLepton::py() const { return py_; }
double MeasuredTauLepton::pz() const { return pz_; }

double MeasuredTauLepton::p() const { return p_; }

int MeasuredTauLepton::decayMode() const { return decayMode_; }

LorentzVector MeasuredTauLepton::p4() const { return p4_; }

Vector MeasuredTauLepton::p3() const { return p3_; }

double MeasuredTauLepton::cosPhi_sinTheta() const { return cosPhi_sinTheta_; }
double MeasuredTauLepton::sinPhi_sinTheta() const { return sinPhi_sinTheta_; }
double MeasuredTauLepton::cosTheta() const { return cosTheta_; }

void MeasuredTauLepton::roundToNdigits(unsigned int nDigis)
{
pt_ = classic_svFit::roundToNdigits(pt_, nDigis);
eta_ = classic_svFit::roundToNdigits(eta_, nDigis);
phi_ = classic_svFit::roundToNdigits(phi_, nDigis);
mass_ = classic_svFit::roundToNdigits(mass_, nDigis);
initialize();
}

void MeasuredTauLepton::initialize()
{
  // CV: relations between pT and p, energy taken from http://en.wikipedia.org/wiki/Pseudorapidity
  p_  = pt_*TMath::CosH(eta_);
  px_ = pt_*TMath::Cos(phi_);
  py_ = pt_*TMath::Sin(phi_);
  pz_ = pt_*TMath::SinH(eta_);
  energy_ = TMath::Sqrt(p_*p_ + preciseVisMass_*preciseVisMass_);
  p4_ = LorentzVector(px_, py_, pz_, energy_);
  p3_ = Vector(px_, py_, pz_);
  double theta = p4_.theta();
  cosPhi_sinTheta_ = TMath::Cos(phi_)*TMath::Sin(theta);
  sinPhi_sinTheta_ = TMath::Sin(phi_)*TMath::Sin(theta);
  cosTheta_ = TMath::Cos(theta);
}