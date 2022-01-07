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
import copy

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

def extrap_full(predicts, rmseps, xmax):
    popt, pcov = curve_fit(line_fit, predicts, rmseps) # Fit a line to all of the unique presmoothed points
    if popt[0]<0:
        print('*****Warning: slope of extrapolation is negative! May be unrealistic. Consider extrapolating from the last local minimum.*****')

    # extrapolate to from the end to xmax
    extrap_predicts = np.array([predicts[-1], xmax])
    rmseps_extrap = popt[0] * extrap_predicts + popt[1]
    rmseps_extrap = rmseps_extrap - rmseps_extrap[0] + rmseps[-1]  # make the line extend from the final point

    # put them together
    extrapolated_rmseps = np.append(rmseps, rmseps_extrap)
    extrapolated_predicts = np.append(predicts, extrap_predicts)
    return extrapolated_rmseps, extrapolated_predicts

def extrap_last_min(predicts, rmseps, xmax):
    extrap = '_extrap_last_min'
    # Find the extrema of the smoothed RMSEPs
    mins = argrelextrema(rmseps, np.less)[0]
    if len(mins) == 0:  # if there is no local min, use the first point as the min
        mins = np.append(mins, [0])

    maxes = argrelextrema(rmseps, np.greater)[0]
    if len(maxes) == 0:  # if there is no local max, then use the last point
        maxes = np.append(maxes, len(rmseps) - 1)
    # if the last local max comes before the last local max, also treat the final values as a local max
    if np.max(mins) > np.max(maxes):
        maxes = np.append(maxes, np.argwhere(rmseps == rmseps[-1])[0])

    # get the points that we will fit a line to
    fit_inds = np.sort(np.array([mins[-1], maxes[-1]]))

    # linear fit to the last local min and local max
    popt, pcov = curve_fit(line_fit, predicts[fit_inds], rmseps[fit_inds])
    if popt[0]<0:
        print('*****Warning: slope of extrapolation is negative! May be unrealistic...*****')

    # calculate the extrapolated values
    extrap_predicts = np.linspace(predicts[-1], xmax, 50)
    rmseps_extrap = popt[0] * extrap_predicts + popt[1]
    rmseps_extrap = rmseps_extrap - rmseps_extrap[0] + rmseps[-1]

    extrapolated_rmseps = np.append(rmseps, rmseps_extrap)
    extrapolated_predicts = np.append(predicts, extrap_predicts)

    return extrapolated_rmseps, extrapolated_predicts

def local_rmse_calc(test_predicts, test_actuals, unk_predicts, windowsize=0.0, min_rmsep_num = 40, sigma = 10, extrapolate = True, full_fit = False,xmax=120):
    test_predicts = np.array(test_predicts)
    test_actuals = np.array(test_actuals)
    test_sq_errs = (test_predicts - test_actuals) ** 2.0

    if min_rmsep_num <= 1 and windowsize == 0:
        print('Window size cannot be zero if min # is <= 1!')
        return
    else:
        # Loop through the dummy predictions, calculating "local" RMSEPs
        dummy_rmseps, dummypredicts = generate_dummy(test_predicts, test_actuals, min_rmsep_num, windowsize, xmax)

        dummy_rmseps_orig = dummy_rmseps  # save a copy of the original dummy rmseps
        dummypredicts_orig = dummypredicts  # save a copy of the original dummy predicts

        if sigma > 0:
            # #Remove duplicate dummy RMSEP values
            dummy_rmseps_presmooth, dummypredicts_presmooth = remove_duplicates(dummy_rmseps_orig, dummypredicts_orig)

            if extrapolate and full_fit:
                dummy_rmseps_presmooth,dummypredicts_presmooth = extrap_full(dummypredicts_presmooth, dummy_rmseps_presmooth, xmax)

            # Re-interpolate (linear) to cover the gaps so that blending works ok
            # Removing the duplicate values and then re-interpolating essentially turns some
            # "stair-steps" in the dummy RMSEPs into linear slopes
            f = interp1d(dummypredicts_presmooth, dummy_rmseps_presmooth)
            dummypredicts_interp = dummypredicts_orig[dummypredicts_orig<np.max(dummypredicts_presmooth)]
            dummy_rmseps_interp = f(dummypredicts_interp)

            #smooth the RMSEPs
            dummy_rmseps_smooth=gaussian_filter1d(dummy_rmseps_interp, sigma, mode='nearest')
            dummypredicts_smooth = dummypredicts_interp
            dummy_rmseps_smooth, dummypredicts_smooth = remove_duplicates(dummy_rmseps_smooth, dummypredicts_smooth)

            if extrapolate and not full_fit:
                dummy_rmseps_extrap, dummypredicts_extrap = extrap_last_min(dummypredicts_smooth, dummy_rmseps_smooth, xmax)

            if not extrapolate:
                dummy_rmseps_extrap = dummy_rmseps_smooth
                dummypredicts_extrap = dummypredicts_smooth

            f_lookup = interp1d(dummypredicts_extrap, dummy_rmseps_extrap)
            unk_predicts_lookup = copy.copy(unk_predicts)
            unk_predicts_lookup[unk_predicts<np.min(dummypredicts_extrap)]=np.min(dummypredicts_extrap)
            unk_predicts_lookup[unk_predicts > np.max(dummypredicts_extrap)] = np.max(dummypredicts_extrap)
            rmseps_lookup = f_lookup(unk_predicts_lookup)

            return rmseps_lookup


def generate_dummy(test_predicts, test_actuals, minval, win, xmax):
    #Create an array of "dummy" predictions
    dummypredicts = np.linspace(0,xmax,num=2000)
    dummy_rmseps = np.zeros_like(dummypredicts)

    test_sq_errs = (test_predicts - test_actuals) ** 2.0

    for j in np.arange(len(dummy_rmseps)):
        rmsep_index = np.abs(test_predicts - dummypredicts[j]) < win
        # ;by default, calculate the RMSEP using test samples with true compositions that are within +/- 10% of the maximum range of the test set
        if np.sum(rmsep_index) >= minval:
            dummy_rmseps[j] = np.sqrt(np.mean(test_sq_errs[rmsep_index]))
        else:
            # If there are fewer samples that meet the above criteria than 10% of the total number of test spectra,
            # then use the nearest # spectra to calculate RMSEP
            if minval > 0:
                if minval == 1:
                    rmsep_index = np.argsort(np.abs(test_predicts - dummypredicts[j]))[0]
                else:
                    rmsep_index = np.argsort(np.abs(test_predicts - dummypredicts[j]))[0:minval - 1]
                dummy_rmseps[j] = np.sqrt(np.mean(test_sq_errs[rmsep_index]))
            else:  # if minval = 0, then fill with nan if there is no data in the window
                dummy_rmseps[j] = np.nan
    return dummy_rmseps, dummypredicts

def local_rmse_explore(test_predicts,test_actuals,windowsize = [0.0], min_rmsep_num = [40], outpath = '', plot_file=None, element='',
                 sigma = None, extrapolate = True, full_fit = False, xmax = 120):
    test_predicts = np.array(test_predicts)
    test_actuals = np.array(test_actuals)
    if not extrapolate:
        print('Not extrapolating. Setting xmax to max of predicts.')
        xmax = np.max(test_predicts)

    extrap = ''
    for minval in min_rmsep_num:
        for win in windowsize:
            if minval <= 1 and win == 0:
                print('Skipping the case where window size = 0 and # of neighbors <= 1')
            else:
                dummy_rmseps,dummypredicts = generate_dummy(test_predicts, test_actuals, minval, win, xmax)

                dummy_rmseps_orig = dummy_rmseps  # save a copy of the original dummy rmseps
                dummypredicts_orig = dummypredicts  # save a copy of the original dummy predicts

                if sigma[0] > 0:
                    # #Remove duplicate dummy RMSEP values
                    dummy_rmseps_presmooth, dummypredicts_presmooth = remove_duplicates(dummy_rmseps_orig, dummypredicts_orig)
                    if plot_file is not None:
                        plot.plot(dummypredicts_orig, dummy_rmseps_orig, linestyle='',marker='.',markersize=2)  # plot the unsmoothed data
                        #mark the points that are not removed as duplicates
                        plot.plot(dummypredicts_presmooth, dummy_rmseps_presmooth, linestyle='',marker='*',markersize=4,color = 'black',label='Unique Values')

                    if extrapolate and full_fit:
                        dummy_rmseps_presmooth, dummypredicts_presmooth = extrap_full(dummypredicts_presmooth, dummy_rmseps_presmooth, xmax)


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
                        dummy_rmseps_extrap = dummy_rmseps_smooth
                        dummypredicts_extrap = dummypredicts_smooth

                        if extrapolate and not full_fit:
                            dummy_rmseps_extrap, dummypredicts_extrap = extrap_last_min(dummypredicts_smooth,
                                                                                        dummy_rmseps_smooth, xmax)

                        if plot_file is not None: #plot each smoothed curve
                            plot.plot(dummypredicts_extrap, dummy_rmseps_extrap, label ='sigma = '+str(s),linestyle='-', marker=None)
                        outdata = pd.DataFrame()
                        outdata['sigma ='+str(s)+'; window ='+str(win)+'; #='+str(minval)] = dummy_rmseps_extrap
                        outdata['prediction'] = dummypredicts_extrap
                        outdata.to_csv(outpath+"/"+"local_rmsep_sigma"+str(s)+'_win'+str(win)+'_n'+str(minval)+'.csv')

               #optionally plot the results
                if plot_file is not None:
                    print('Plotting:')
                    print('Window: ' + str(win) + '; Min #: ' + str(minval) + '; Sigma: ' + str(sigma))
                    if sigma is None:
                        plot.plot(dummypredicts,dummy_rmseps,linestyle='-',marker = 'o',label='Window = '+str(round(win,1))+'; min = '+str(round(minval,1)))
                        outdata = pd.DataFrame()
                        outdata['prediction'] = dummypredicts
                        outdata['window ='+str(win)+'; #='+str(minval)] = dummy_rmseps
                        outdata.to_csv(outpath+"/"+"local_rmsep_win" + str(win) + '_n' + str(minval) + '.csv')

                    ax = plot.gca()
                    ax.set_xlim([0, np.min([xmax,2*np.max(test_actuals)])])
                    ax.set_ylim([0,2*np.max(dummy_rmseps_orig)])
                    ax.set_title(element+' - Window = '+str(round(win,2))+'; Min # = '+str(minval))


                    try:
                        plot.legend()
                    except:
                        pass
                    ax.set_xlabel('Prediction')
                    ax.set_ylabel('Local RMSEP')
                    plot.tight_layout()

                    plot.savefig(outpath+"/"+plot_file.split('.')[0]+'_'+element+'_win'+str(round(win,2))+'_min'+str(minval)+extrap+'.png',dpi=1000,figsize=(4,3))
                    plot.close()


