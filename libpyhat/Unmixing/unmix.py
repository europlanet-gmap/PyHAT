import numpy as np
from pysptools.abundance_maps import NNLS,FCLS,UCLS
#import MESMA
import pandas as pd


def unmix(data, endmembers, normalize, mask, col = 'wvl', unmix_method = 'NNLS',):
    supported_methods = ("NNLS","FCLS","UCLS")#,"MESMA")
    try:
        if unmix_method.upper() in supported_methods:
            if unmix_method == 'NNLS':
                method = NNLS()
            if unmix_method == 'FCLS':
                method = FCLS()
            if unmix_method == 'UCLS':
                method = UCLS()
            # if unmix_method == "MESMA":
            #     method = MESMA()

        else:
            print(f"{unmix_method} is not a supported method.  Supported methods are {supported_methods}")
            return 1
    except KeyError:
        print(f"Unable to instantiate class from {unmix_method}.")
        return 1

    spectra = data[col].to_numpy()
    if len(spectra.shape) == 2:
        spectra = np.expand_dims(spectra, 0)
    try:
        em_names = endmembers[('meta','endmember_name')]
    except:
        print("No column labeled ('meta','endmember_name') found! Assigning numerical names:")
        em_names = ['EM'+str(i+1) for i in range(endmembers.shape[0])]
        print(em_names)
        endmembers[('meta','endmember_name')] = em_names
    em_array = endmembers[col].to_numpy()
    results = method.map(spectra, em_array,normalize=normalize, mask=mask)
    results = pd.DataFrame(np.squeeze(results),columns=pd.MultiIndex.from_tuples([(unmix_method,i) for i in em_names]))
    data = pd.concat([data,results],axis=1)
    return data
