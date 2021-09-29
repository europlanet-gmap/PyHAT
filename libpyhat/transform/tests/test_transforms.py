import numpy as np
import pandas as pd
from libpyhat.examples import get_path
import libpyhat.transform.deriv as deriv
import libpyhat.transform.dim_red as dim_red
import libpyhat.transform.interp as interp
import libpyhat.transform.mask as mask
import libpyhat.transform.meancenter as meancenter
import libpyhat.transform.multiply_vector as multiply_vector
import libpyhat.transform.norm as norm
import libpyhat.transform.shift_spect as shift_spect
import libpyhat.clustering.cluster as cluster
from libpyhat.transform.dim_reductions.mnf import MNF

np.random.seed(1)

def test_shift_spect():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    result = shift_spect.shift_spect(df, -1.0)
    expected = [898.64928571, 973.62444444, 1034.46444444, 1004.54, 939.16222222]
    np.testing.assert_array_almost_equal(expected, np.array(result['wvl'].iloc[0, 0:5]))
    assert result[('meta', 'Shift')].shape == (103,)


def test_norm():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    result = norm.norm(df, [[580, 590], [590, 600]], col_var='wvl')
    np.testing.assert_almost_equal(result['wvl'].iloc[0, :].sum(), 2.0)


def test_multiply_vector():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    result = multiply_vector.multiply_vector(df, get_path('vector.csv'))
    expected = [1646.12, 1548.12, 1656.12, 1656.12, 1732.12]
    np.testing.assert_array_almost_equal(expected, np.array(result['wvl'].iloc[0, 0:5]))

    result = multiply_vector.multiply_vector(df, get_path('bad_vector.csv'))
    assert result == 0


def test_meancenter():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    result_df, mean_vect = meancenter.meancenter(df, 'wvl')
    expected = [-168.05398058, 579.71601942, 309.16601942, 709.21601942, -341.00398058]
    expected_mv = [991.11398058, 1160.24990291, 1287.87126214, 931.56058252, 838.89067961]
    np.testing.assert_array_almost_equal(expected, np.array(result_df['wvl'].iloc[0:5, 0]))
    np.testing.assert_array_almost_equal(expected_mv, np.array(mean_vect)[0:5])

    # test providing the mean vector
    mean_vect.iloc[:] = 1
    result_df2, mean_vect2 = meancenter.meancenter(df, 'wvl', previous_mean=mean_vect)
    expected2 = np.array(expected) - 1.0
    expected_mv2 = [1., 1., 1., 1., 1.]
    np.testing.assert_array_almost_equal(expected2, np.array(result_df2['wvl'].iloc[0:5, 0]))
    np.testing.assert_array_almost_equal(expected_mv2, np.array(mean_vect2)[0:5])

    # test mismatched wvls
    mean_vect.index = np.array(mean_vect.index, dtype=float) + 1.0
    result = meancenter.meancenter(df, 'wvl', previous_mean=mean_vect)
    assert result == 0


def test_mask():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    result = mask.mask(df, get_path('mask.csv'))
    assert result['wvl'].columns[0] == 586.049
    assert result['wvl'].columns[-1] == 589.869
    assert result['wvl'].shape == (103, 18)
    assert result['masked'].shape == (103, 26)


def test_interp():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    result = interp.interp(df, [588, 590, 592, 594])
    expected = [1637.58, 1104.47964286, 830.53321429, 857.77875]
    assert result['wvl'].shape == (103, 4)
    np.testing.assert_array_almost_equal(expected, np.array(result['wvl'].iloc[0, :]))


def test_deriv():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    result = deriv.deriv(df)
    expected = [-0.08370717, 1.08648488, 0.83536337, 1.59556113, 0.13666476]
    np.testing.assert_array_almost_equal(expected, np.array(result['wvl'].iloc[0:5, 0]))


def test_dimred_JADE():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])

    params = {'n_components': 3}

    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'JADE-ICA', [], params)
    expected_loadings = [0.56247385, 0.19292341, 3.42289881]
    expected_scores = [174708.34499912, 125682.55985134, 145155.40758151]

    assert df['JADE-ICA (wvl)'].shape == (103, 3)
    np.testing.assert_almost_equal(expected_loadings, np.squeeze(np.array(dimred_obj.ica_jade_loadings[:, 0])))
    np.testing.assert_array_almost_equal(expected_scores, np.array(df['JADE-ICA (wvl)'].iloc[0, :]))


def test_dimred_LLE():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])

    params = {'n_components': 3,
              'n_neighbors': 10,
              'reg': 1e-3}
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'LLE', [], params)
    expected_err = 2.0687806439705738e-05
    expected_scores = [0.11088153, 0.01215013, -0.03551393]

    assert df['LLE (wvl)'].shape == (103, 3)
    np.testing.assert_almost_equal(expected_err, dimred_obj.reconstruction_error_)
    np.testing.assert_array_almost_equal(np.abs(expected_scores), np.abs(np.array(df['LLE (wvl)'].iloc[0, :])),
                                         decimal=4)


def test_dimred_tSNE():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])

    params = {
        'n_components': 2,
        'learning_rate': 200.0,
        'n_iter': 1000,
        'n_iter_without_progress': 300,
        'perplexity': 30,
        'init': 'pca'}
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 't-SNE', [], params)
    expected_div = 0.38829776644706726
    expected_scores = [9938.469727, -802.161682]

    assert df['t-SNE (wvl)'].shape == (103, 2)
    np.testing.assert_almost_equal(expected_div, dimred_obj.kl_divergence_)
    np.testing.assert_array_almost_equal(expected_scores, np.array(df['t-SNE (wvl)'].iloc[0, :]))


def test_dimred_FastICA():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])

    params = {'n_components': 3, 'random_state': 1}
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'FastICA', [], params)
    expected_comps = [-2.190278e-05, 1.498101e-06, 9.082887e-07]
    expected_scores = [0.03252833, -0.03749623, -0.11434307]

    assert df['FastICA (wvl)'].shape == (103, 3)
    np.testing.assert_array_almost_equal(np.sort(expected_comps), np.sort(dimred_obj.components_[:, 0]), decimal=5)
    np.testing.assert_array_almost_equal(np.sort(expected_scores), np.sort(np.array(df['FastICA (wvl)'].iloc[0, :])),
                                         decimal=5)


def test_dimred_PCA():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])

    params = {'n_components': 3}
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'PCA', [], params)
    expected_expl_var = [0.96051211, 0.01683739, 0.01471955]
    expected_scores = [10092.96265442, -628.16699776, -359.06894452]
    assert df['PCA (wvl)'].shape == (103, 3)
    np.testing.assert_array_almost_equal(expected_expl_var, dimred_obj.explained_variance_ratio_)
    np.testing.assert_array_almost_equal(expected_scores, np.array(df['PCA (wvl)'].iloc[0, :]))


def test_dimred_NNMF():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    df['wvl'] = df['wvl'] - 1000  # make some values negative to test adding a constant
    dim_red.check_positive(df['wvl'])
    params = {'n_components': 3,
              'random_state': 0,
              'add_constant': True}
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'NNMF', [], params)
    expected_comps = [10.27191532, 34.62489686, 3.06822373]
    expected_scores = [49.42458628, 3.9910722, 27.03100371]
    assert df['NNMF (wvl)'].shape == (103, 3)
    np.testing.assert_array_almost_equal(expected_comps, dimred_obj.components_[:, 0])
    np.testing.assert_array_almost_equal(expected_scores, np.array(df['NNMF (wvl)'].iloc[0, :]))


def test_dimred_NNMF_usingLIBS():
    '''Tests the LDA function using real world labeled LIBS data and
    physically/chemically intuitive tests.'''

    # Open the test dataset, which contains LIBS library spectra
    df = pd.read_csv(get_path('labeled_LIBS_testfile.csv'), header=[0, 1])

    # NMF requires all positive values, let's set the floor to 0
    df['wvl'] = np.where(df['wvl'].values < 0, 0, df['wvl'].values)

    # Set up the parameters for the NMF algorithm
    # The number of components and iterations are increased because NMF
    # doesn't tend to converge nicely with this dataset. Might need
    # more andesite samples to get it behave or extend the number of
    # wavelengths that are fed to it.
    # NMF won't run in PyHAT without the add_constant param defined
    params = {}
    kws = {'add_constant': False, 'n_components': 2, 'max_iter': 20000}

    # Run NMF
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'NNMF', params=params, kws=kws, ycol='Geologic name')

    # Find the indicies that correspond to the spectra and NMF results
    # for the two labeled rock types
    ind_bas = np.where(df['Geologic name'].values == 'Basalt')[0]
    ind_and = np.where(df['Geologic name'].values == 'Andesite')[0]

    # Simple test is to check the centers of the clusters. They should
    # be distinct enough, and this was verified visually in writing the test.
    and_center = np.mean(np.array([df['NNMF (wvl)']['NNMF-1'].values[ind_and], df['NNMF (wvl)']['NNMF-2'].values[ind_and]]),
                         axis=1)
    bas_center = np.mean(np.array([df['NNMF (wvl)']['NNMF-1'].values[ind_bas], df['NNMF (wvl)']['NNMF-2'].values[ind_bas]]),
                         axis=1)
    np.testing.assert_almost_equal(np.abs(bas_center - and_center), [1316662.64123436, 1196221.16644762])

    # Also, let's make sure to do a simple check to make sure
    # the clusters are well seperated (by 2 standard deviations of their
    # average standard deviation along the two components).
    stds = np.mean(np.std(df['NNMF (wvl)'].values[ind_bas], axis=0)) + np.mean(
        np.std(df['NNMF (wvl)'].values[ind_and], axis=0))
    dist = np.linalg.norm(np.vstack([and_center, bas_center]))
    np.testing.assert_array_less(np.array([2 * stds]), np.array([dist]))


def test_dimred_LDA():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])

    kws = {'n_clusters': 5,
           'n_init': 10,
           'max_iter': 100,
           'tol': 0.01,
           'random_state': 1}
    cluster.cluster(df, 'wvl', 'K-Means', [], kws)
    params = {'n_components': 3}
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'LDA', [], params, ycol='K-Means')
    expected_coefs = np.sort([0.02209121, 0.0016516, 0.01139357, 0.06448139, 0.07085655])
    expected_scores = np.sort([11.89340048, 0.41598425, 0.22964169])
    assert df['LDA (wvl)'].shape == (103, 3)
    # +/- sign and order of these values can vary, use sort and abs to stabilize things
    np.testing.assert_array_almost_equal(expected_coefs, np.sort(np.abs(dimred_obj.coef_[:, 0])))
    np.testing.assert_array_almost_equal(expected_scores, np.sort(np.abs(np.array(df['LDA (wvl)'].iloc[0, :]))))


def test_dimred_LDA_usingLIBS():
    '''Tests the LDA function using real world labeled LIBS data and 
    physically/chemically intuitive tests.'''
    
    #Open the test dataset, which contains LIBS library spectra
    df = pd.read_csv(get_path('labeled_LIBS_testfile.csv'), header=[0, 1])
    
    #Set up the parameters for the LDA algorithm
    #There are only two labels/categories 'Andesite' and 'Basalt'
    #So there can only be a single component
    params = {}
    kws    = {'n_components': 1}
    
    #If the basalt and andesite spectra are distinct from one another
    #LDA should should present us with distinct clusters
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'LDA', params=params, kws=kws, ycol='Geologic name')
    
    #Find the indicies that correspond to the spectra and LDA results
    #for the two labeled rock types
    ind_bas = np.where(df['Geologic name'].values=='Basalt')[0]
    ind_and = np.where(df['Geologic name'].values=='Andesite')[0]
    
    #Simple test is to check the distance of the center of the clusters.
    dist = np.abs(np.median(df['LDA']['LDA-1'].values[ind_bas]) - np.median(df['LDA']['LDA-1'].values[ind_and]))
    np.testing.assert_almost_equal(dist, 5.367776196776056)
    
    #Also, let's make sure to do a simple check to make sure
    #the clusters are well seperated (by 2 standard deviations).
    stds = np.std(df['LDA']['LDA-1'].values[ind_bas]) + np.std(df['LDA']['LDA-1'].values[ind_and])
    np.testing.assert_array_less(np.array([2*stds]), np.array([dist]))


def test_dimred_LDA_usingSalinas():
    '''Tests the LDA function using real world labeled Salinas data and
    intuitive tests.'''
    
    #Open the test dataset, which contains Salinas library spectra
    df = pd.read_csv(get_path('labeled_Salinas_testfile.csv'), header=[0])
    
    #Set up the parameters for the LDA algorithm. There are only
    #two labels that we're looking at, so there can only be a single component
    params = {}
    kws    = {'n_components': 1}
    
    #Grab the columns that contain spectral data
    cols = list(df.columns[4:-1])
    
    #LDA should should present us with distinct clusters with respect to the labels
    df, dimred_obj = dim_red.dim_red(df, cols, 'LDA', params=params, kws=kws, ycol='gt')
    
    #Find the rows that correspond to the ground truth ('gt' column) with
    #vegetation type 2 and 6
    ind_2 = np.where(df['gt'].values==2)[0]
    ind_6 = np.where(df['gt'].values==6)[0]
    
    #Simple test is to check the distance of the center of the clusters.
    dist = np.abs(np.median(df.iloc[:,-1].values[ind_2]) - np.median(df.iloc[:,-1].values[ind_6]))
    np.testing.assert_almost_equal(dist, 4.293384107071617)
    
    #Also, let's make sure to do a simple check to make sure
    #the clusters are well seperated (by 2 standard deviations).
    stds = np.std(df.iloc[:,-1].values[ind_2]) + np.std(df.iloc[:,-1].values[ind_6])
    np.testing.assert_array_less(np.array([2*stds]), np.array([dist]))


def test_dimred_MNF():
    df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    params = {'n_components': 4}
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'MNF', [], params)
    score_result = np.sort(np.array(df['MNF (wvl)'].iloc[0, :]))
    expected_scores = [-36.6691721, -5.29645881, -3.63660052, 598.27972428]
    np.testing.assert_array_almost_equal(expected_scores, score_result)

    mnf = MNF()
    x = np.array(df['wvl'])
    try:
        comps = mnf.fit_transform('foo')  # test the case where the wrong type of data is passed

    except:
        try:
            comps = mnf.fit_transform(x.T)  # test the case where # of wvls is > # of samples
        except:

            comps = mnf.fit_transform(x)  # test the case where a numpy array is passed
            score_result = np.sort(np.sort(comps[0, :]))
            expected_scores = [-36.6691721, -5.29645881, -3.63660052, 598.27972428]
            np.testing.assert_array_almost_equal(expected_scores, score_result)


def test_dimred_MNF_usingLIBS():
    '''Tests the MNF function using real world labeled LIBS data and
    with physically/chemically intuitive tests.'''
    
    #Open the test dataset, which contains LIBS library spectra
    df = pd.read_csv(get_path('labeled_LIBS_testfile.csv'), header=[0, 1])
    
    #Set up parameters and arguments for MNF
    params = {}
    kws    = {'n_components':2}
    
    #Grab the andesite data
    ind_and = np.where(df['Geologic name'].values=='Andesite')[0]
    
    #Run MNF on a single type of sample, so it can determine the
    #channels for that spectra, which should be relatively consistent,
    #experimental factors held constant (assumption).
    df, dimred_obj = dim_red.dim_red(df.loc[ind_and], 'wvl', 'MNF',  params=params, kws=kws)
    
    #Using the MNF transform, let's ask it to give us the signal
    #and noise channels for the first andesite spectrum
    x1 = dimred_obj.fit_transform(df['wvl'].values.T)[:,0] #first component (signal)
    x2 = dimred_obj.fit_transform(df['wvl'].values.T)[:,1] #second component (noise)
    
    #Now let's check that there's good correlation with the data
    #and the signal channel, but poor correlation with the noise channel
    from scipy.stats import pearsonr
    r = pearsonr(df['wvl'].values.T[:,0], x1)
    
    #Let's make sure this is near 1
    np.testing.assert_array_almost_equal(np.array(r), [0.99724445, 0.        ])
    
    #Let's also make sure the noise channel has poor correlation
    r = pearsonr(df['wvl'].values.T[:,0], x2)
    np.testing.assert_array_almost_equal(np.array(r), [0.05240176, 0.24263178])


def test_dimred_MNF_usingSalinas():
    '''Tests the MNF function using real world labeled Salinas
     data and intuitive tests.'''
    
    #Open the test dataset, which contains labeled Salinas spectra
    df = pd.read_csv(get_path('labeled_Salinas_testfile.csv'), header=[0])
    
    #Set up parameters and arguments for MNF
    params = {}
    kws    = {'n_components':2}
    
    #Grab the data for vegetation type 2, which is in the ground truth
    #('gt') column
    ind_2 = np.where(df['gt'].values==2)[0]
    
    #Run MNF on a single type of sample, so it can determine the
    #channels for that spectra, which should be relatively consistent,
    #experimental factors held constant (assumption).
    cols = list(df.columns[4:-1])
    df, dimred_obj = dim_red.dim_red(df.loc[ind_2], cols, 'MNF',  params=params, kws=kws)
    
    #Using the MNF transform, let's ask it to give us the signal
    #and noise channels for the first spectrum for type 2
    x1 = dimred_obj.fit_transform(df[cols].values.T)[:,0] #first component (signal)
    x2 = dimred_obj.fit_transform(df[cols].values.T)[:,1] #second component (noise)
    
    #Now let's check that there's good correlation with the data
    #and the signal channel, but poor correlation with the noise channel
    from scipy.stats import pearsonr
    r = pearsonr(df[cols].values.T[:,0], x1)
    
    #Let's make sure this is near 1
    np.testing.assert_array_almost_equal(np.array(r), [0.99724445, 0.        ])
    
    #Let's also make sure the noise channel has poor correlation
    r = pearsonr(df[cols].values.T[:,0], x2)
    np.testing.assert_array_almost_equal(np.array(r), [0.05240176, 0.24263178])


def test_dimred_LFDA():
    # df = pd.read_csv(get_path('test_data.csv'), header=[0, 1])
    df = pd.read_csv(get_path('iris.csv'))
    params = {'r': 3, 'metric': 'plain', 'knn': 5}
    cols = list(df.columns[1:5])
    df, dimred_obj = dim_red.dim_red(df, cols, 'LFDA', [], params, ycol='Species')

    expected_Z = [[1.17658471, -5.73187812, -0.70134324], [1.26725406, -5.23944184, -0.84506262]]
    expected_Tr = [[0.15532635, -0.67208433, -0.47664679], [-0.24346924, -0.71603884, 0.47809747]]
    expected_values = [-0.70134324, -0.84506262, -0.64763099, -0.66074162, -0.60586882]
    np.testing.assert_array_almost_equal(expected_Z, dimred_obj.Z[0:2, :])
    np.testing.assert_array_almost_equal(expected_Tr, dimred_obj.Tr[0:2, :])
    np.testing.assert_array_almost_equal(expected_values, df.iloc[0:5, -1])

    df = pd.read_csv(get_path('iris.csv'))
    params = {'r': 3, 'metric': 'weighted', 'knn': 5}
    cols = list(df.columns[1:5])
    df, dimred_obj = dim_red.dim_red(df, cols, 'LFDA', [], params, ycol='Species')

    expected_Z = [[40.24100588, -39.75445661, -2.99501843], [43.34203702, -36.33907748, -3.60875811]]
    expected_Tr = [[5.3124001, -4.66135999, -2.03547398], [-8.32702232, -4.96621426, 2.04166896]]
    expected_values = [-2.99501843, -3.60875811, -2.76564543, -2.82163311, -2.58730414]
    np.testing.assert_array_almost_equal(expected_Z, dimred_obj.Z[0:2, :])
    np.testing.assert_array_almost_equal(expected_Tr, dimred_obj.Tr[0:2, :])
    np.testing.assert_array_almost_equal(expected_values, df.iloc[0:5, -1])

    df = pd.read_csv(get_path('iris.csv'))
    params = {'r': 3, 'metric': 'orthonormalized', 'knn': 5}
    cols = list(df.columns[1:5])
    df, dimred_obj = dim_red.dim_red(df, cols, 'LFDA', [], params, ycol='Species')

    expected_Z = [[-1.17658471, -6.20333752, 0.5894996], [-1.26725406, -5.71840297, 0.76816463]]
    expected_Tr = [[-0.15532635, -0.73171269, 0.46949654], [0.24346924, -0.67718402, -0.54512867]]
    expected_values = [0.5894996, 0.76816463, 0.54826003, 0.58978237, 0.48803708]
    np.testing.assert_array_almost_equal(expected_Z, dimred_obj.Z[0:2, :])
    np.testing.assert_array_almost_equal(expected_Tr, dimred_obj.Tr[0:2, :])
    np.testing.assert_array_almost_equal(expected_values, df.iloc[0:5, -1])


def test_dimred_LFDA_usingLIBS():
    '''Tests the LFDA function using real world labeled LIBS data and
    with physically/chemically intuitive tests. 
    
    Note: Tried developing tests for array equivalence, but LFDA is rather 
    inconsistent in how it chooses the location of the two clusters
    and their absolute locations. For now, the test is only for separability'''
    
    #Open the test dataset, which contains LIBS library spectra
    df = pd.read_csv(get_path('labeled_LIBS_testfile.csv'), header=[0, 1])
    
    #Set up parameters and arguments for LFDA. For a super simple test,
    #we can use a single dimension to verify that LFDA can seperate between
    #the basalt and andesite labels. We just assume a single cluster in 
    #the local space (knn)
    params = {}
    kws    = {'r':1, 'metric':'plain', 'knn':1}
    
    #Perform LFDA
    df, dimred_obj = dim_red.dim_red(df, 'wvl', 'LFDA', params=params, kws=kws, ycol='Geologic name')
    
    #Grab the indicies of the two labels
    ind_bas = np.where(df['Geologic name'].values=='Basalt')[0]
    ind_and = np.where(df['Geologic name'].values=='Andesite')[0]
    
    #Compute the mean and standard deviation of the two sample types
    c1_bas = df['LFDA (wvl)']['LFDA-1'].values[ind_bas]
    c1_and = df['LFDA (wvl)']['LFDA-1'].values[ind_and]
    m1_bas = np.mean(c1_bas)
    m1_and = np.mean(c1_and)
    s1_bas_and = np.std(c1_bas)+np.std(c1_and)
    
    #Verify that there are two *very* seperable clusters in component 1
    #by comparing their distance to 10000 times their combined standard deviations
    np.testing.assert_array_less(s1_bas_and*1000, np.abs(m1_bas-m1_and))


def test_dimred_LFDA_usingSalinas():
    '''Tests the LFDA function using real world labeled Salinas data and
    with intuitive tests. 
    
    Note: Tried developing tests for array equivalence, but LFDA is rather 
    inconsistent in how it chooses the location of the two clusters
    and their absolute locations. For now, the test is only for separability'''
    
    #Open the test dataset, which contains labeled Salinas spectra
    df = pd.read_csv(get_path('labeled_Salinas_testfile.csv'), header=[0])
    
    #Find the rows that correspond to the ground truth ('gt' column) with
    #vegetation type 2 and 6
    ind_2 = np.where(df['gt'].values==2)[0]
    ind_6 = np.where(df['gt'].values==6)[0]
    ind_2_6 = np.concatenate([ind_2, ind_6])
    
    #Let's only use the rows with types 2 and 6
    df = df.loc[ind_2_6]
    
    #Set up parameters and arguments for LFDA. For a super simple test,
    #we can use a single dimension to verify that LFDA can seperate between
    #label 2 and 6, which are distinct vegetation type.
    #We just assume a single cluster in the local space (knn)
    params = {}
    kws    = {'r':1, 'metric':'plain', 'knn':1}
    
    #Perform LFDA
    cols = list(df.columns[4:-1])
    df, dimred_obj = dim_red.dim_red(df, cols, 'LFDA', params=params, kws=kws, ycol='gt')
    
    #Compute the mean and standard deviation of the two sample types
    #First, find their indicies in the new dataframe
    ind_2 = np.where(df['gt'].values==2)[0]
    ind_6 = np.where(df['gt'].values==6)[0]
    c1_2 = df.iloc[:,-1].values[ind_2]
    c1_6 = df.iloc[:,-1].values[ind_6]
    m1_2 = np.mean(c1_2)
    m1_6 = np.mean(c1_6)
    s1_2_6 = np.std(c1_2)+np.std(c1_6)
    
    #Verify that there are two *very* seperable clusters in component 1
    #by comparing their distance to 10000 times their combined standard deviations
    np.testing.assert_array_less(s1_2_6*1000, np.abs(m1_2 - m1_6))
