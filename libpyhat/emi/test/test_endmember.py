import numpy as np
import pandas as pd
from libpyhat.examples import get_path
from libpyhat import emi

def test_endmember_PPI_usingSalinas():
    '''Intuitive tests the PPI function using real world labeled Salinas data.
    
    Note: The underlying Salinas dataset contains several labels ('ground truths')
    that correspond to specific crop types:
    gt=0  Area in-between crop fields; can have a huge range of spectral characteristics
    gt=9  FILL IN
    gt=10 FILL IN
    gt=14 FILL IN'''
    
    ## Open the test dataset, which contains Salinas spectra
    # Find the path to the image file
    fp   = get_path('labeled_SalinasImage_testfile.tif')
    # Open the tif with gdal, and turn it into an array
    img  = gdal.Open(filepath)
    data = img.ReadAsArray()
    # Unravel the image into a row of pixels (2-D array)
    data = np.reshape(data.T, (75*16, 205))
    
    # The first band is the groundtruth (terrain type)
    gt   = data[:,0]
    # The subsequent bands are spectral intensities
    data = data[:,1:]
    
    # Build a pandas dataframe with appropriate 2-level
    # multiindex column structure that PyHAT expects
    df = pd.DataFrame(data, columns=list(np.arange(0,np.shape(data)[1])))
    df.columns = pd.MultiIndex.from_tuples(zip(['wvl']*np.shape(data)[1], df.columns))
    
    # Run PPI with 3 endmembers specified
    x,y = emi.emi(df, col='wvl', emi_method='PPI', n_endmembers=3)
    
    # ENVI's PPI function found the spectrum at 1187 (gt=14) 
    # to be the first endmember during testing.
    np.testing.assert_equal(y[0], 1187)
    
    # Pysptools' PPI function in testing found two unique ground
    # truths (gt=9 and gt=10) to be endmembers, though these did 
    # not match to the second and third endmembers found in ENVI's 
    # PPI function.
    np.testing.assert_qual(gt[y[1:]], [9, 10])
