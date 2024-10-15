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

    