from bin.jade import JADE
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF
from libpyhat.transform.dim_reductions.mnf import MNF
from libpyhat.transform.dim_reductions.lfda import LFDA
import numpy as np
#This function does dimensionality reduction on a data frame full of spectra. A number of different methos can be chosen

def dim_red(df, xcol, method, params, kws, load_fit=None, ycol=None):
    xdata = df[xcol]

    if method == 'PCA':
        do_dim_red = PCA(*params, **kws)
    if method == 'FastICA':
        do_dim_red = FastICA(*params, **kws)
    if method == 't-SNE':
        do_dim_red = TSNE(*params, **kws)
    if method == 'LLE':
        do_dim_red = LocallyLinearEmbedding(*params, **kws)
    if method == 'JADE-ICA':
        do_dim_red = JADE(*params, **kws)
    if method == 'LDA':
        do_dim_red = LinearDiscriminantAnalysis(*params, **kws)
    if method == 'NNMF':
        add_const = kws.pop('add_constant')
        do_dim_red = NMF(*params, **kws)
    if method == 'MNF':
        do_dim_red = MNF(*params, **kws)
    if method == 'LFDA':
        do_dim_red = LFDA(*params, **kws)

    if load_fit:
        do_dim_red = load_fit
    else:
        if method not in ['t-SNE','MNF']:
            if ycol is not None:
                #find the multi-index that matches the specified single index
                ycol_tuple = [a for a in df.columns.values if ycol in a][0]
                ydata = df[ycol_tuple]
                if method == 'LDA':
                    #Check to make sure # of components isn't too high for LDA
                    max_nc = np.min([len(np.unique(ydata))-1,len(df[xcol].columns)])
                    if do_dim_red.n_components > max_nc:
                       print("n_components cannot be larger than min(n_features, n_classes - 1)")
                       print('n_features = '+str(len(df[xcol].columns)))
                       print('n_classes-1 = '+str(len(np.unique(ydata))-1))
                       print("Setting n_components from "+str(do_dim_red.n_components)+" to "+str(max_nc))
                       do_dim_red.n_components = max_nc
                do_dim_red.fit(xdata,ydata)
            else:
                if method == 'NNMF':
                    if add_const:
                        if xdata.min().min()<0:
                            xdata = xdata-xdata.min().min()
                        else:
                            print('Data is already positive: no need to add a constant!')
                    check_positive(xdata)
                do_dim_red.fit(xdata)
            dim_red_result = do_dim_red.transform(xdata)
        else:
            if method == 't-SNE':
                dim_red_result = do_dim_red.fit_transform(xdata)
            if method == 'MNF':
                dim_red_result = do_dim_red.fit_transform(xdata)

    for i in list(range(1, dim_red_result.shape[
                               1] + 1)):  # will need to revisit this for other methods that don't use n_components to make sure column names still mamke sense
        df[(method+' ('+str(xcol)+')', method+'-'+str(i))] = dim_red_result[:, i - 1]

    return df, do_dim_red

def check_positive(data):
    if data.min().min()<0:
        print('NNMF will not work with data containing negative values!')
