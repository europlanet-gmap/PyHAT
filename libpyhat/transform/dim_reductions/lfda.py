"""Translated from https://github.com/cran/lfda/blob/master/R/lfda.R
#' Local Fisher Discriminant Analysis for
#' Supervised Dimensionality Reduction
#'
#' Performs local fisher discriminant analysis (LFDA) on the given data.
#'
#' LFDA is a method for linear dimensionality reduction that maximizes
#' between-class scatter and minimizes within-class scatter while at the
#' same time maintain the local structure of the data so that multimodal
#' data can be embedded appropriately. Its limitation is that it only
#' looks for linear boundaries between clusters. In this case, a non-linear
#' version called kernel LFDA will be used instead. Three metric types can
#' be used if needed.
#'
#' x = n x d matrix of original samples.
#'          n is the number of samples.
#' y = length n vector of class labels
#' r = dimensionality of reduced space (default: d)
#' metric = type of metric in the embedding space (no default)
#'               'weighted'        --- weighted eigenvectors
#'               'orthonormalized' --- orthonormalized
#'               'plain'           --- raw eigenvectors
#' knn = parameter used in local scaling method (default: 5)
#'
#' Returns:
#' T = d x r transformation matrix (Z = x * T)
#' Z = n x r matrix of dimensionality reduced samples
#'
#' Keywords: lfda local fisher discriminant transformation mahalanobis metric
#'
#' @author Yuan Tang
#' @seealso See \code{\link{klfda}} for the kernelized variant of
#'          LFDA (Kernel LFDA).
#'
#' References:
#' Sugiyama, M (2007).
#' Dimensionality reduction of multimodal labeled data by
#' local Fisher discriminant analysis.
#' \emph{Journal of Machine Learning Research}, vol.\bold{8}, 1027--1061.
#'
#' Sugiyama, M (2006).
#' Local Fisher discriminant analysis for supervised dimensionality reduction.
#' In W. W. Cohen and A. Moore (Eds.), \emph{Proceedings of 23rd International
#' Conference on Machine Learning (ICML2006)}, 905--912.
#'
"""
import numpy as np
import warnings
#' Get Affinity Matrix
#'
#' This function returns an affinity matrix within knn-nearest neighbors from the distance matrix.
#'
#' distance2  =The distance matrix for each observation
#' knn = The number of nearest neighbors
#' nc = The number of observations for data in this class
# returns:
# A = an affinity matrix - the larger the element in the matrix, the closer two data points are

def getAffinity(distance2, knn, nc):
    sortarr = np.sort(distance2,axis=0)
    if sortarr.shape[1] < knn + 1:
        print("knn is too large, please try to reduce it.")
        return
    kNNdist2 = sortarr[knn,:]
    sigma = np.sqrt(kNNdist2)
    sigma = np.expand_dims(sigma, axis=0)
    localscale = np.dot(sigma.T,sigma)
    flag = localscale != 0
    A = np.zeros((nc,nc))
    A[flag] = np.exp(-distance2[flag]/localscale[flag])
    return A

class LFDA():
    def __init__(self, r = None, metric = 'plain', knn = 5):
        self.metric = metric
        self.knn = knn
        self.r = r

    def fit(self, x, y):
        # metric can be: "orthonormalized", "plain", "weighted"
        x = np.array(x).T
        y = np.array(y).T
        d = np.size(x,0) # number of rows
        n = np.size(x,1) # number of columns

        # if no dimension reduction requested, set r to d
        if self.r == None:
            self.r = d

        tSb = np.zeros((d, d))
        tSw = np.zeros((d, d))

        # compute the optimal scatter matrices in a classwise manner
        for value in np.unique(y.flatten()):
            Xc = x[:, y == value]
            nc = np.size(Xc,1) # number of columns

            # # determine local scaling for locality-preserving projection
            Xc2 = np.array(np.sum(np.power(Xc, 2),axis=0))

            # # calculate the distance, using a self-defined repmat function that's the same
            # # as repmat() in Matlab
            Xc2tile = np.tile(Xc2, (nc,1))
            distance2 = Xc2tile + Xc2tile.T-2*np.dot(Xc.T, Xc)

            # # Get affinity matrix
            A = getAffinity(distance2, self.knn, nc)

            Xc1 = np.expand_dims(np.array(Xc.sum(axis=1)),axis=1)

            A_tiled = np.tile(A.sum(axis=0), (d, 1)).T
            G = np.dot(Xc,A_tiled*Xc.T) - np.dot(Xc,np.dot(A,Xc.T))
            tSb = tSb + (G / n) + np.dot(Xc, Xc.T) * (1 - nc / n) + np.dot(Xc1,(Xc1.T/n))
            tSw = tSw + G / nc

        X1 = np.expand_dims(np.sum(x,axis=1),axis=0)
        tSb = tSb - np.dot((X1.T/n),X1) - tSw
        tSb = (tSb + tSb.T) / 2
        tSw = (tSw + tSw.T) / 2



         # find generalized eigenvalues and normalized eigenvectors of the problem
        eigVal, eigVec = np.linalg.eig(np.dot(np.linalg.inv(tSw), tSb))
        if self.r == d:
            # without dimensionality reduction
            pass
        else:
            # dimensionality reduction (select only the r largest eigenvalues of the problem)
            eigVal = eigVal[0:self.r]
            eigVec = eigVec[:,0:self.r]

        pass
        # Based on metric return other values
        # options to require a particular type of returned transform matrix

        if self.metric == "orthonormalized":
            Tr = np.linalg.qr(eigVec)[0]
        elif self.metric == "weighted":
            Tr = eigVec*np.tile(np.sqrt(eigVal).T, (d, 1))
        elif self.metric == "plain":
            Tr = eigVec
        else:
            "Invalid Metric Type. Using 'plain'"
            Tr = eigVec

        Z = np.dot(Tr.T, x).T
        
        if(np.any(np.iscomplex(Z))):
            warnings.warn('The returned matricies are complex! The numpy.linalg.eig function within the lfda implementation is likely the culprit.')

        self.Tr = Tr
        self.Z = Z

        return

    # ' LFDA Transformation/Prediction on New Data
    # '
    # ' This function transforms a data set, usually a testing set, using the trained LFDA metric
    # ' newdata = The data to be transformed
    def transform(self, newdata=None):
        return np.dot(newdata, self.Tr)
