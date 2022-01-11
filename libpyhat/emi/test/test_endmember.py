import numpy as np
import pandas as pd
from libpyhat.examples import get_path
from libpyhat.emi import emi
from osgeo import gdal

def test_endmember_PPI_usingSalinas():
    '''Intuitive tests the PPI function using real world labeled 
    Salinas data.
    
    Note: The underlying Salinas dataset contains several labels 
    ('ground truths') that correspond to specific crop types:
    gt=0  Area in-between crop fields; can have a huge range of spectral characteristics
    gt=9  soil-vineyard-develop
    gt=10 corn-senesced-weeds
    gt=14 lettuce-romaine-7wk
    
    Note: We're reading in 205 spectral bands and do not include the 
    band location as a parameter.'''
    
    ## Open the test dataset, which contains Salinas spectra
    # Find the path to the image file
    fp   = get_path('labeled_SalinasImage_testfile.tif')
    # Open the tif with gdal, and turn it into an array
    img  = gdal.Open(fp)
    data = img.ReadAsArray()
    
    # Unraveling the image data into a row of pixels
    # (2-D array) the stupid way. This is guaranteed to
    # preserve order in a predictable way.
    d = np.zeros((np.shape(data)[1]*np.shape(data)[2], np.shape(data)[0]))
    k = 0
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            d[k, :] = data[:,i,j]
            k += 1    
    
    # Seperate out ground truth from data
    gt = d[:, 0]
    d  = d[:, 1:]
    
    # Build a pandas dataframe with appropriate 2-level
    # multiindex column structure that PyHAT expects
    df = pd.DataFrame(d, columns=list(np.arange(0,np.shape(d)[1])))
    df.columns = pd.MultiIndex.from_tuples(zip(['wvl']*np.shape(d)[1], df.columns))
    
    # Run PPI with 3 endmembers specified
    x,y = emi.emi(df, col='wvl', emi_method='PPI', n_endmembers=3)
    
    # Sort the endmembers and check that they match testing
    # Ordering of the endmembers seems to be consistent so
    # they're not sorted.
    # Note: ENVI's PPI function found gt=0, gt=0, and gt=14.
    np.testing.assert_equal(gt[y], [0, 10, 10])

def test_endmember_FIPPI_usingSalinas():
    '''Intuitive tests the FIPPI function using real world labeled 
    Salinas data.
    
    Note: The underlying Salinas dataset contains several labels 
    ('ground truths') that correspond to specific crop types:
    gt=0  Area in-between crop fields; can have a huge range of spectral characteristics
    gt=9  soil-vineyard-develop
    gt=10 corn-senesced-weeds
    gt=14 lettuce-romaine-7wk
    
    Note: We're reading in 205 spectral bands and do not include the 
    band location as a parameter.
    
    Note: ENVI's FIPPI function found the same spectral endmembers 
    as ENVI's PPI function.
    '''
    
    ## Open the test dataset, which contains Salinas spectra
    # Find the path to the image file
    fp   = get_path('labeled_SalinasImage_testfile.tif')
    # Open the tif with gdal, and turn it into an array
    img  = gdal.Open(fp)
    data = img.ReadAsArray()
    
    # Unraveling the image data into a row of pixels
    # (2-D array) the stupid way. This is guaranteed to
    # preserve order in a predictable way.
    d = np.zeros((np.shape(data)[1]*np.shape(data)[2], np.shape(data)[0]))
    k = 0
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            d[k, :] = data[:,i,j]
            k += 1    
    
    # Seperate out ground truth from data
    gt = d[:, 0]
    d  = d[:, 1:]
        
    # Build a pandas dataframe with appropriate 2-level
    # multiindex column structure that PyHAT expects
    df = pd.DataFrame(d, columns=list(np.arange(0,np.shape(d)[1])))
    df.columns = pd.MultiIndex.from_tuples(zip(['wvl']*np.shape(d)[1], df.columns))
    
    # Run FIPPI with 3 endmembers specified
    # Note: For whatever reason, it will produce 5 end members anyway...
    x,y = emi.emi(df, col='wvl', emi_method='FIPPI', n_endmembers=3)
    
    # In testing, Pysptools found 3 different versions of the soil
    # gt (0) and two of the lettuce/romaine.
    # Ordering of the endmembers seems to be consistent so
    # they're not sorted.
    # FIPPI results in ENVI matched those of PPI in ENVI (gt=0, 0, and 14).
    np.testing.assert_equal(gt[y], [14, 14, 0, 0, 0])

def test_endmember_NFINDR_usingSalinas():
    '''Intuitive tests the N-FINDR function using real world labeled 
    Salinas data.
    
    Note: The underlying Salinas dataset contains several labels 
    ('ground truths') that correspond to specific crop types:
    gt=0  Area in-between crop fields; can have a huge range of spectral characteristics
    gt=9  soil-vineyard-develop
    gt=10 corn-senesced-weeds
    gt=14 lettuce-romaine-7wk
    
    Note: We're reading in 205 spectral bands and do not include the 
    band location as a parameter.
    '''
    
    ## Open the test dataset, which contains Salinas spectra
    # Find the path to the image file
    fp   = get_path('labeled_SalinasImage_testfile.tif')
    # Open the tif with gdal, and turn it into an array
    img  = gdal.Open(fp)
    data = img.ReadAsArray()
      
    # Unraveling the image data into a row of pixels
    # (2-D array) the stupid way. This is guaranteed to
    # preserve order in a predictable way.
    d = np.zeros((np.shape(data)[1]*np.shape(data)[2], np.shape(data)[0]))
    k = 0
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            d[k, :] = data[:,i,j]
            k += 1
    
    # Seperate out ground truth from data
    gt = d[:, 0]
    d  = d[:, 1:]
    
    # Build a pandas dataframe with appropriate 2-level
    # multiindex column structure that PyHAT expects
    df = pd.DataFrame(d, columns=list(np.arange(0,np.shape(d)[1])))
    df.columns = pd.MultiIndex.from_tuples(zip(['wvl']*np.shape(d)[1], df.columns))
    
    # Run N-FINDR with 3 endmembers specified
    x,y = emi.emi(df, col='wvl', emi_method='N-FINDR', n_endmembers=3)
    
    # In testing, Pysptools' N-FINDR found gt=0, gt=0, and gt=14.
    # Ordering of the endmembers is not consistent from run to run,
    # so sorting is necessary.
    np.testing.assert_equal(np.sort(gt[y]), [0, 0, 14])

def test_endmember_ATGP_usingSalinas():
    '''Intuitive tests the ATGP function using real world labeled 
    Salinas data.
    
    Note: The Salinas dataset contains several labels 
    ('ground truths') that correspond to specific crop types:
    gt=0  Area in-between crop fields; can have a huge range of spectral characteristics
    gt=9  soil-vineyard-develop
    gt=10 corn-senesced-weeds
    gt=14 lettuce-romaine-7wk
    
    Note: We're reading in 205 spectral bands and do not include the 
    band location as a parameter.
    '''
    
    ## Open the test dataset, which contains Salinas spectra
    # Find the path to the image file
    fp   = get_path('labeled_SalinasImage_testfile.tif')
    # Open the tif with gdal, and turn it into an array
    img  = gdal.Open(fp)
    data = img.ReadAsArray()
    
    # Unraveling the image data into a row of pixels
    # (2-D array) the stupid way. This is guaranteed to
    # preserve order in a predictable way.
    d = np.zeros((np.shape(data)[1]*np.shape(data)[2], np.shape(data)[0]))
    k = 0
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            d[k, :] = data[:,i,j]
            k += 1
    
    # Seperate out ground truth from data
    gt = d[:, 0]
    d  = d[:, 1:]
    
    # Build a pandas dataframe with appropriate 2-level
    # multiindex column structure that PyHAT expects
    df = pd.DataFrame(d, columns=list(np.arange(0,np.shape(d)[1])))
    df.columns = pd.MultiIndex.from_tuples(zip(['wvl']*np.shape(d)[1], df.columns))
    
    # Run ATGP with 3 endmembers specified
    x,y = emi.emi(df, col='wvl', emi_method='ATGP', n_endmembers=3)
    
    # In testing, Pysptools' ATGP found gt=0, gt=14, and gt=0.
    # Ordering of the endmembers seems to be consistent from run to run,
    # so we do not sort.
    np.testing.assert_equal(gt[y], [0, 14, 0])
    
    # Note: ENVI's ATGP found exact matches to these endmembers.
    np.testing.assert_equal(y, [568, 8, 147])
