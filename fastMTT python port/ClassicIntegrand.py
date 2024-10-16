import numpy as np
import SVfitIntegratorMarkovChain as imc

###Ustawienie zmiennej środowiskowej USE_SVFITTF
import os
USE_SVFITTF = os.getenv('USE_SVFITTF', 'False') == 'True'

def norm(v):
    return np.sqrt(v.mag2())

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compCrossProduct(v1, v2):
    return np.cross(v1, v2)

class ClassicSVFitIntegrand:
    def __init__(self, verbosity, USE_SVFITTF = False):
        ###
        #CONSTANTS DEFINITION
        ###
        self.verbosity = verbosity
        if self.verbosity:
            print("ClassicSVFitIntegrand::ClassicSVFitIntegrand()\n")

        #self.gSVfitIntegrand = self ### Jest coś takiego w kodzie, ale zobaczymy czy to potrzebne
        self.xMin = []
        self.xMax = []
    
    ###Destruktor, jeśli chcemy

    def setVerbosity(self, verbosity):
        self.verbosity = verbosity
    
    def addLogM_fixed(self, value, power):
        pass
    #prosty warning // do przepisania

    #Do sprawdzenia -- kod wygląda raczej prosto, ale pisał go copilot, więc trzeba przejrzeć
    def addLogM_dynamic(self, value, power):
        self.addLogM_dynamic_ = value
        if self.addLogM_dynamic_:
            if power != "":
                power_tstring = power.replace("m", "x").replace("mass", "x")
                formula_name = "ClassicSVfitIntegrand_addLogM_dynamic_formula"
                self.addLogM_dynamic_formula = self.create_formula(formula_name, power_tstring)
            else:
                print(f"Warning: expression = '{power}' is invalid --> disabling dynamic logM term !!")
                self.addLogM_dynamic = False

        if self.addLogM_dynamic_ and self.addLogM_fixed_:
            print("Warning: simultaneous use of fixed and dynamic logM terms not supported --> disabling fixed logM term !!")
            self.addLogM_fixed = False

    def setDiTauMassConstraint(self, diTauMass):
        self.diTauMassConstraint = diTauMass
    
    def setHistogramAdapter(self, histogramAdapter):
        self.histogramAdapter = histogramAdapter

    def setLegIntegrationParams(self, iLeg, aParams):
        self.legIntegrationParams[iLeg] = aParams

    def setNumDimensions(self, numDimensions):
        self.numDimensions = numDimensions
        
    def setIntegrationRanges(self, xl, xu):
        for iDimension in range(self.numDimensions):
            if len(self.xMin) <= iDimension:
                self.xMin.append(xl[iDimension])
            else:
                self.xMin[iDimension] = xl[iDimension]
            if len(self.xMax) <= iDimension:
                self.xMax.append(xu[iDimension])
            else:
                self.xMax[iDimension] = xu[iDimension]

    if USE_SVFITTF:
        #Tu się zastanowić co z clone. Skąd je wziąć i czy trzeba definiować kolejną klasę?
        def setHadTauTF(self, hadTauTF):
            self.hadTauTF1 = hadTauTF.Clone("leg1")
            self.hadTauTF2 = hadTauTF.Clone("leg2")

        def enableHadTauTF(self):
            if not self.hadTauTF1 or not self.hadTauTF2:
                print("No tau pT transfer functions defined, call 'setHadTauTF' function first !!")
                assert(0)
            self.useHadTauTF = True

        def disableHadTauTF(self):
            self.useHadTauTF = False

        def setRhoHadTau(self, rhoHadTau):
            self.rhoHadTau = rhoHadTau

    def setLeptonInputs(self, measuredTauLeptons):
        if self.verbosity >= 2:
            print("<ClassicSVfitIntegrand::setInputs>:")

        #reset 'LeptonNumber' and 'MatrixInversion' error codes
        self.errorCode &= (self.errorCode ^ self.LeptonNumber)
        self.errorCode &= (self.errorCode ^ self.MatrixInversion)

        if len(measuredTauLeptons) != 2:
            print("Error: Number of MeasuredTauLeptons is not equal to two !!")
            self.errorCode |= self.LeptonNumber
        self.measuredTauLepton1 = measuredTauLeptons[0]
        self.leg1isLep = measuredTauLepton1.type == MeasuredTauLepton.kTauToElecDecay or measuredTauLepton1.type == MeasuredTauLepton.kTauToMuDecay
        self.leg1Mass = measuredTauLepton1.mass
        self.leg1Mass2 = self.leg1Mass**2
        eZ1 = normalize(measuredTauLepton1.p3())
        eY1 = normalize(compCrossProduct(eZ1, beamAxis))
        eX1 = normalize(compCrossProduct(eY1, eZ1))

        if self.verbosity >= 2:
            print(f"eX1: theta = {eX1.theta()}, phi = {eX1.phi()}, norm = {norm(eX1)}")
            print(f"eY1: theta = {eY1.theta()}, phi = {eY1.phi()}, norm = {norm(eY1)}")
            print(f"eZ1: theta = {eZ1.theta()}, phi = {eZ1.phi()}, norm = {norm(eZ1)}")
            print(f"(eX1 x eY1 = {norm(compCrossProduct(eX1, eY1))}, eX1 x eZ1 = {norm(compCrossProduct(eY1, eZ1))}, eY1 x eZ1 = {norm(compCrossProduct(eY1, eZ1))})")

        self.leg1eX_x = eX1.x
        self.leg1eX_y = eX1.y
        self.leg1eX_z = eX1.z
        self.leg1eY_x = eY1.x
        self.leg1eY_y = eY1.y
        self.leg1eY_z = eY1.z
        self.leg1eZ_x = eZ1.x
        self.leg1eZ_y = eZ1.y
        self.leg1eZ_z = eZ1.z

        self.measuredTauLepton2 = measuredTauLeptons[1]
        self.leg2isLep = measuredTauLepton2.type == MeasuredTauLepton.kTauToElecDecay or measuredTauLepton2.type == MeasuredTauLepton.kTauToMuDecay
        leg2Mass = measuredTauLepton2.mass
        leg2Mass2 = leg2Mass**2
        eZ2 = normalize(measuredTauLepton2.p3())
        eY2 = normalize(compCrossProduct(eZ2, beamAxis))
        eX2 = normalize(compCrossProduct(eY2, eZ2))

        if self.verbosity >= 2:
            print(f"eX2: theta = {eX2.theta()}, phi = {eX2.phi()}, norm = {norm(eX2)}")
            print(f"eY2: theta = {eY2.theta()}, phi = {eY2.phi()}, norm = {norm(eY2)}")
            print(f"eZ2: theta = {eZ2.theta()}, phi = {eZ2.phi()}, norm = {norm(eZ2)}")
        
        self.leg2eX_x = eX2.x
        self.leg2eX_y = eX2.y
        self.leg2eX_z = eX2.z
        self.leg2eY_x = eY2.x
        self.leg2eY_y = eY2.y
        self.leg2eY_z = eY2.z
        self.leg2eZ_x = eZ2.x
        self.leg2eZ_y = eZ2.y
        self.leg2eZ_z = eZ2.z

        self.mVis_measured = (measuredTauLepton1.p4 + measuredTauLepton2.p4).mass
        if self.verbosity >= 2:
            print(f"mVis = {mVis_measured}")
        self.mVis2_measured = mVis_measured**2

        self.phaseSpaceComponentCache = 0

        if USE_SVFITTF:
            if self.useHadTauTF:
                if measuredTauLepton1.type == MeasuredTauLepton.kTauToHadDecay:
                    assert hadTauTF1
                    hadTauTF1.setDecayMode(measuredTauLepton1.decayMode)
                if measuredTauLepton2.type == MeasuredTauLepton.kTauToHadDecay:
                    assert hadTauTF2
                    hadTauTF2.setDecayMode(measuredTauLepton2.decayMode)
                    
    ###Tu kontynuować (lin. 233 oryginalnego kodu)

    



