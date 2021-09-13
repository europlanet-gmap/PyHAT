"""
;+
;This function is used to calculate the RMSEP on-the-fly as a function of predicted composition.
;It uses the test set results to create a plot of RMSEP vs composition. This is then smoothed and extrapolated
;and re-sampled to be used as a look-up table for the predictions of unknown targets.
;
;Inputs:
;       predicts = Array containing the predicted compositions that need RMSEPs
;       test_predicts = Hash containing the predicted compositions for the test set
;       test_actuals = Hash containing the actual compositions for the test set
;       elems = string array containing major oxide names
;       makeplot = Optional keyword that produces plots of RMSEP vs composition
;Outputs:
;       rmseps = Array of RMSEP values calculated for each of the predictions in "predicts"
;-
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

def line_fit(x, A, B):
    return A*x + B

def remove_duplicates(rmseps, predicts):
# #Remove duplicate dummy RMSEP values
    rmseps, uniq_ind = np.unique(np.round(rmseps,5), return_index=True)
    predicts=predicts[uniq_ind]

    # #re-sort the dummy predicts and rmseps
    rmseps = rmseps[np.argsort(predicts)]
    predicts = np.sort(predicts)

    return rmseps, predicts

def dynamic_rmse(test_predicts,test_actuals,windowsize = [0.0], min_rmsep_num = [40], plot_file=None, element='',
                 sigma = None, extrapolate = True, full_fit = False, xmax = 120):
    test_predicts = np.array(test_predicts)
    test_actuals = np.array(test_actuals)

    # calculate the squared errors
    test_sq_errs=(test_predicts-test_actuals)**2.0

    #Create an array of "dummy" predictions
    dummypredicts = np.linspace(0,xmax,num=2000)
    dummy_rmseps = np.zeros_like(dummypredicts)

    if plot_file is not None:
        figsize = (3*len(windowsize),2*len(min_rmsep_num))
        if len(min_rmsep_num)>len(windowsize):
            ncols = len(min_rmsep_num)
            nrows = len(windowsize)
        else:
            ncols = len(windowsize)
            nrows = len(min_rmsep_num)
        fig, axes = plot.subplots(nrows=nrows,ncols=ncols,figsize=figsize,sharex='all',sharey='all')
        try:
            axes.shape
            axes = axes.flat
        except:
            axes = [axes]
        plot.subplots_adjust(wspace=0, hspace=0)
    i = 0
    for minval in min_rmsep_num:
        for win in windowsize:
            if minval == 0 and win == 0:
                print('Skipping the case where window size = 0 and # of neighbors = 1')
            else:
                #Loop through the dummy predictions, calculating "local" RMSEPs
                for j in np.arange(len(dummy_rmseps)):
                    rmsep_index = np.abs(test_predicts - dummypredicts[j]) < win
                    #;by default, calculate the RMSEP using test samples with true compositions that are within +/- 10% of the maximum range of the test set
                    if np.sum(rmsep_index) >= minval:
                        dummy_rmseps[j]=np.sqrt(np.mean(test_sq_errs[rmsep_index]))
                    else:
                        #If there are fewer samples that meet the above criteria than 10% of the total number of test spectra,
                        # then use the nearest # spectra to calculate RMSEP
                        if minval > 0:
                            if minval == 1:
                                rmsep_index = np.argsort(np.abs(test_predicts - dummypredicts[j]))[0]
                            else:
                                rmsep_index=np.argsort(np.abs(test_predicts-dummypredicts[j]))[0:minval-1]
                            dummy_rmseps[j]=np.sqrt(np.mean(test_sq_errs[rmsep_index]))
                        else: #if minval = 0, then fill with nan if there is no data in the window
                            dummy_rmseps[j] = np.nan

                dummy_rmseps_orig = dummy_rmseps  # save a copy of the original dummy rmseps
                dummypredicts_orig = dummypredicts  # save a copy of the original dummy predicts

                if sigma[0] > 0:
                    # #Remove duplicate dummy RMSEP values
                    dummy_rmseps_presmooth, dummypredicts_presmooth = remove_duplicates(dummy_rmseps_orig, dummypredicts_orig)
                    if plot_file is not None:
                        axes[i].plot(dummypredicts_orig, dummy_rmseps_orig, linestyle='-', marker='o')  # plot the unsmoothed data
                        #mark the points that are not removed as duplicates
                        axes[i].plot(dummypredicts_presmooth, dummy_rmseps_presmooth, linestyle='-',marker='*',markersize=4,color = 'black')

                    if full_fit:
                        popt, pcov = curve_fit(line_fit, dummypredicts_presmooth, # Fit a line to all of the unique presmoothed points
                                               dummy_rmseps_presmooth)
                        #extrapolate to from the end to xmax
                        extrap_predicts = np.array([dummypredicts_presmooth[-1],xmax])
                        rmseps_extrap = popt[0] * extrap_predicts + popt[1]
                        rmseps_extrap = rmseps_extrap - rmseps_extrap[0] + dummy_rmseps_presmooth[-1] #make the line extend from the final point

                        #put them together
                        dummy_rmseps_presmooth = np.append(dummy_rmseps_presmooth, rmseps_extrap)
                        dummypredicts_presmooth = np.append(dummypredicts_presmooth, extrap_predicts)

                    # Re-interpolate (linear) to cover the gaps so that blending works ok
                    # Removing the duplicate values and then re-interpolating essentially turns some
                    # "stair-steps" in the dummy RMSEPs into linear slopes
                    f = interp1d(dummypredicts_presmooth, dummy_rmseps_presmooth)
                    dummypredicts_interp = dummypredicts_orig[dummypredicts_orig<np.max(dummypredicts_presmooth)]
                    dummy_rmseps_interp = f(dummypredicts_interp)

                    for s in sigma:
                        #smooth the RMSEPs
                        dummy_rmseps_smooth=gaussian_filter1d(dummy_rmseps_interp,s, mode='nearest')
                        dummypredicts_smooth = dummypredicts_interp
                        dummy_rmseps_smooth, dummypredicts_smooth = remove_duplicates(dummy_rmseps_smooth, dummypredicts_smooth)

                        if extrapolate and not full_fit:
                            #Find the extrema of the smoothed RMSEPs
                            mins = argrelextrema(dummy_rmseps_smooth,np.less)[0]
                            if len(mins) == 0: #if there is no local min, use the first point as the min
                                mins = np.append(mins, [0])

                            maxes = argrelextrema(dummy_rmseps_smooth,np.greater)[0]
                            if len(maxes) == 0: #if there is no local max, then use the last point
                                maxes = np.append(maxes,len(dummy_rmseps_smooth)-1)
                            # if the last local max comes before the last local max, also treat the final values as a local max
                            if np.max(mins) > np.max(maxes):
                                maxes = np.append(maxes, np.argwhere(dummy_rmseps_smooth == dummy_rmseps_smooth[-1])[0])

                            #get the points that we will fit a line to
                            fit_inds = np.sort(np.array([mins[-1],maxes[-1]]))

                            #linear fit to the last local min and local max
                            popt, pcov = curve_fit(line_fit,dummypredicts_smooth[fit_inds], dummy_rmseps_smooth[fit_inds])

                            #calculate the extrapolated values
                            extrap_predicts = np.linspace(dummypredicts_smooth[-1],xmax,50)
                            rmseps_extrap = popt[0] * extrap_predicts + popt[1]
                            rmseps_extrap = rmseps_extrap - rmseps_extrap[0] + dummy_rmseps_smooth[-1]

                            dummy_rmseps_extrap = np.append(dummy_rmseps_smooth,rmseps_extrap)
                            dummypredicts_extrap = np.append(dummypredicts_smooth,extrap_predicts)
                        else:
                            dummy_rmseps_extrap = dummy_rmseps_smooth
                            dummypredicts_extrap = dummypredicts_smooth

                        if plot_file is not None: #plot each smoothed curve
                            axes[i].plot(dummypredicts_extrap, dummy_rmseps_extrap, label ='sigma = '+str(s),linestyle='-', marker=None)
                        outdata = pd.DataFrame()
                        outdata['sigma ='+str(s)+'; window ='+str(win)+'; #='+str(minval)] = dummy_rmseps_extrap
                        outdata['prediction'] = dummypredicts_extrap
                        outdata.to_csv("local_rmsep_sigma"+str(s)+'_win'+str(win)+'_n'+str(minval)+'.csv')

               #optionally plot the results
            if plot_file is not None:
                if sigma is None:
                    axes[i].plot(dummypredicts,dummy_rmseps,linestyle='-',marker = 'o',label='Window = '+str(round(win,1))+'; min = '+str(round(minval,1)))
                    outdata = pd.DataFrame()
                    outdata['prediction'] = dummypredicts
                    outdata['window ='+str(win)+'; #='+str(minval)] = dummy_rmseps
                    outdata.to_csv("local_rmsep_win" + str(win) + '_n' + str(minval) + '.csv')

                # # Look up the expected RMSEP for the actual predictions
            #    for j=0,n_elements(predicts[i,*])-1 do begin
            #        !null = min(abs(predicts[i,j]-dummypredicts_resamp), imin)   ;Find index of minimum value
            #        rmseps[i,j]=dummy_rmseps_resamp[imin]

                axes[i].set_xlim([0, 2*np.max(test_actuals)])
                axes[i].set_ylim([0,2*np.max(dummy_rmseps_orig)])
                axes[i].set_title('Window = '+str(round(win,2))+'; Min # = '+str(minval))

                i=i+1
    try:
        axes[-1].legend()
    except:
        pass
    fig.supxlabel('Prediction')
    fig.supylabel('Local RMSEP')
    plot.suptitle(element)
    plot.tight_layout()
    plot.savefig(plot_file,dpi=1000)
    plot.close()

