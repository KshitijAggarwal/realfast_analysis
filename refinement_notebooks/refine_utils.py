# # Updated for rfpipe version 1.3.1
import numpy as np
import logging
logger = logging.getLogger('rfpipe')
from matplotlib import gridspec
import pylab as plt
import matplotlib

params = {
        'axes.labelsize' : 14,
        'font.size' : 9,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': False,
        'figure.figsize': [12, 10]
        }
matplotlib.rcParams.update(params)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def disperse(data, dm, freqs, inttime):
    from rfpipe import util
    from rfpipe.search import dedisperse_roll
    delay = util.calc_delay(freqs, freqs.max(), dm, inttime)
    return dedisperse_roll(data, delay)

def cd_refine(cd, nsubbands = 4, save = False, devicenum='0', mode='GPU', outdir=None):
    from rfpipe.search import make_dmt
    segment, candint, dmind, dtind, beamnum = cd.loc
    st = cd.state
    width_m = st.dtarr[dtind]
    timewindow = st.prefs.timewindow
    tsamp = st.inttime*width_m
    dm = st.dmarr[dmind]
    ft_dedisp = np.flip((cd.data.real.sum(axis=2).T), axis=0)
    chan_freqs = np.flip(st.freq*1000, axis=0)  # from high to low, MHz
    nf, nt = np.shape(ft_dedisp)
    
    candloc = cd.loc
    
    logger.debug('Size of the FT array is ({0}, {1})'.format(nf, nt))

    try:
        assert nt > 0
    except AssertionError as err:
        logger.exception("Number of time bins is equal to 0")
        raise err

    try:
        assert nf > 0
    except AssertionError as err:
        logger.exception("Number of frequency bins is equal to 0")
        raise err    

    roll_to_center = nt//2 - cd.integration_rel
    ft_dedisp = np.roll(ft_dedisp, shift=roll_to_center, axis=1)

    # If timewindow is not set during search, set it equal to the number of time bins of candidate
    if nt != timewindow:
        logger.info('Setting timewindow equal to nt = {0}'.format(nt))
        timewindow = nt
    else:
        logger.info('Timewindow length is {0}'.format(timewindow))

    try:
        assert nf == len(chan_freqs)
    except AssertionError as err:
        logger.exception("Number of frequency channel in data should match the frequency list")
        raise err

    if dm is not 0:
        dm_start = 0
        dm_end = 2*dm
    else:
        dm_start = -10
        dm_end = 10

    logger.info('Generating DM-time for candid {0} in DM range {1:.2f}--{2:.2f} pc/cm3'
                .format(cd.candid, dm_start, dm_end))

    logger.info("Using gpu devicenum: {0}".format(devicenum))
    os.environ['CUDA_VISIBLE_DEVICES'] = devicenum

    dmt = make_dmt(ft_dedisp, dm_start-dm, dm_end-dm, 256, chan_freqs/1000,
                   tsamp, mode=mode, devicenum=int(devicenum))
    
    dispersed = disperse(ft_dedisp, -1*dm, chan_freqs/1000, tsamp)
    
    subsnrs, subts, bands = calc_subband_info(ft_dedisp, chan_freqs, nsubbands)    
    logging.info(f'Generating time series of full band')
    ts_full = ft_dedisp.sum(0)
    logging.info(f'Calculating SNR of full band')
    snr_full = calc_snr(ts_full)

    to_print = []
    logging.info(f'candloc: {candloc}, dm: {dm:.2f}')
    to_print.append(f'candloc: {candloc}, dm: {dm:.2f}\n')
    logging.info(f'SNR of full band is: {snr_full:.2f}')
    to_print.append(f'SNR of full band is: {snr_full:.2f}\n')
    logging.info(f'Subbanded SNRs are:')    
    to_print.append(f'Subbanded SNRs are:\n')
    for i in range(nsubbands):
        logging.info(f'Band: {chan_freqs[bands[i][0]]:.2f}-{chan_freqs[bands[i][1]-1]:.2f}, SNR: {subsnrs[i]:.2f}')
        to_print.append(f'Band: {chan_freqs[bands[i][0]]:.2f}-{chan_freqs[bands[i][1]-1]:.2f}, SNR: {subsnrs[i]:.2f}\n')

    str_print = ''.join(to_print)

    ts = np.arange(timewindow)*tsamp
        
    gs = gridspec.GridSpec(4, 3, width_ratios=[4, 0.1, 2], height_ratios=[1, 1, 1, 1], wspace=0.02, hspace=0.15)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[2, 0])
    ax4 = plt.subplot(gs[3, 0])
    ax11 = plt.subplot(gs[0, 1])
    ax22 = plt.subplot(gs[1, 1])
    ax33 = plt.subplot(gs[2, 1])
    ax44 = plt.subplot(gs[3, 1])
    ax5 = plt.subplot(gs[:, 2])

    x_loc = 0.1
    y_loc = 0.5

    for i in range(nsubbands):
        ax1.plot(ts, subts[i] - subts[i].mean(), label = f'Band: {chan_freqs[bands[i][0]]:.0f}-{chan_freqs[bands[i][1]-1]:.0f}')
    ax1.plot(ts, subts.sum(0) - subts.sum(0).mean(), 'k.', label = 'Full Band')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3, fancybox=True, shadow=True)
    ax1.set_ylabel('Flux (Arb. units)')
    ax1.set_xlim(np.min(ts), np.max(ts))
    ax11.text(x_loc, y_loc, 'Time Series', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax11.axis('off')

    ax2.imshow(ft_dedisp, aspect='auto', extent=[ts[0], ts[-1], np.min(chan_freqs), np.max(chan_freqs)])
    ax2.set_ylabel('Freq')
    ax22.text(x_loc, y_loc, 'Dedispersed FT', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax22.axis('off')

    ax3.imshow(dispersed, aspect='auto', extent=[ts[0], ts[-1], np.min(chan_freqs), np.max(chan_freqs)])
    ax3.set_ylabel('Freq')
    ax33.text(x_loc, y_loc, 'Original dispersed FT', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax33.axis('off')

    ax4.imshow(dmt, aspect='auto', extent=[ts[0], ts[-1], dm+1*dm, dm-dm])
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('DM')
    ax44.text(x_loc, y_loc, 'DM-Time', fontsize=14, ha='center', va='center', wrap=True, rotation=-90)
    ax44.axis('off')

    ax5.text(0.02, 0.8, str_print, fontsize=14, ha='left', va='top', wrap=True)
    ax5.axis('off')
    segment, candint, dmind, dtind, beamnum = candloc
    #plt.tight_layout()
    if save==True:
        if outdir:
            plt.savefig(outdir+'{0}_refined.png'.format(cd.candid), bbox_inches='tight')
        else:
            plt.savefig('{0}_refined.png'.format(cd.candid), bbox_inches='tight')

    plt.show()

def calc_subband_info(ft, chan_freqs, nsubbands=4):
    nf, nt = ft.shape

    subbandsize = nf//nsubbands
    bandstarts = np.arange(1,nf,subbandsize) - 1
    subsnrs = np.zeros(nsubbands)
    subts = np.zeros((nsubbands, ft.shape[1]))
    bands = []
    for i in range(nsubbands):
        bandstart = i*subbandsize
        if i == nsubbands-1:
            bandend = nf-1
        else:
            bandend = (i+1)*subbandsize

        bands.append([bandstart, bandend])
        logging.info(f'Generating time series of band: {chan_freqs[bands[i][0]]:.0f}-{chan_freqs[bands[i][1]-1]:.0f}')
        subts[i, :] = ft[bandstart: bandend,:].sum(0)
        logging.info(f'Calculating SNR of band: {chan_freqs[bands[i][0]]:.0f}-{chan_freqs[bands[i][1]-1]:.0f}')
        subsnrs[i] = calc_snr(subts[i, :])
    return subsnrs, subts, bands
    
def madtostd(array):
    return 1.4826*np.median(np.abs(array-np.median(array)))

def calc_snr(ts):
    std =  madtostd(ts)
    if std == 0:
        logging.warning('Standard Deviation of time series is 0. SNR not defined.')
        snr = np.nan
        return snr

    noise_mask = (np.median(ts) - 3*std < ts) & (ts < np.median(ts) + 3*std)
    if noise_mask.sum() == len(ts):
        logging.warning('Time series is just noise, SNR = 0.')
        snr = 0
    else:
        mean_ts = np.mean(ts[noise_mask])
        std = madtostd(ts[noise_mask]-mean_ts)
        if std == 0:
            logging.warning('Noise Standard Deviation is 0. SNR not defined.')
        snr = np.max(ts[~noise_mask]-mean_ts)/std
    return snr

def max_timewindow(st):
    if st.prefs.maxdm is None:
        maxdm = 1000
    else:
        maxdm = st.prefs.maxdm
    return int(4148808.0 * maxdm * (1 / np.min(st.freq) ** 2 - 1 / np.max(st.freq) ** 2) / 1000 / 10**6 // st.inttime)