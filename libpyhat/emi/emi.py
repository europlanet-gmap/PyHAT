import numpy as np
from pysptools.eea import PPI, FIPPI, NFINDR, ATGP
from spectral.algorithms import smacc

class SMACC():
    def __init__(self):
        self.max_residual_norm = float('Inf')
    def extract(self, spectra, n_endmembers):
        spect_scaled = spectra/np.max(spectra)  #smacc documentation suggests scaling large values for numerical stability
        self.S,self.F,self.R = smacc(np.squeeze(spect_scaled),min_endmembers=n_endmembers,max_residual_norm=float('Inf'))



def emi(data, col = 'wvl', emi_method = 'FIPPI', n_endmembers = 6):
    supported_methods = ("FIPPI", "PPI", "N-FINDR", "ATGP","SMACC")
    try:
        if emi_method.upper() in supported_methods:
            if emi_method == 'PPI':
                method = PPI()
            if emi_method == 'FIPPI':
                method = FIPPI()
            if emi_method == 'N-FINDR':
                method = NFINDR()
            if emi_method == 'ATGP':
                method = ATGP()
            if emi_method == "SMACC":
                method = SMACC()

        else:
            print(f"{emi_method} is not a supported method.  Supported methods are {supported_methods}")
            return 1
    except KeyError:
        print(f"Unable to instantiate class from {emi_method}.")
        return 1

    spectra = data[col].to_numpy()
    if len(spectra.shape) == 2:
        spectra = np.expand_dims(spectra, 0)

    method.extract(spectra, n_endmembers)
    if emi_method == "SMACC":
        endmember_indices = np.any(method.F == 1, axis=1)
    else:
        endmember_indices = method.idx

    indices = np.zeros(spectra.shape[1], dtype=int)
    indices[endmember_indices] = 1
    data[("endmembers", emi_method)] = indices
    return data, endmember_indices
