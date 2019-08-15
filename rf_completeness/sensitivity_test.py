import rfpipe
from rfpipe import util

import numpy as np 
from astropy import time
import pylab as plt
import pandas as pd
from random import randint, uniform
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.propagate = False

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

t0 = time.Time.now().mjd
meta = rfpipe.metadata.mock_metadata(t0,t0+2/(24*3600), 27, 4, 32*4, 4, 5e3, datasource='sim', antconfig='B')

prefs = {'dtarr': [1], 'npix_max': 1024,'maxdm': 3000, 'applyonlineflags': False, 'flaglist': [],
         'clustercands': False, 'fftmode': 'cuda'}

st = rfpipe.state.State(inmeta = meta, inprefs=prefs)
data = rfpipe.source.read_segment(st,0)

params=[]
trials = 100000
visualize_cluster = False
for trial in range(trials):
    snr = uniform(3,30)
    mock = rfpipe.util.make_transient_params(st, snr = snr, data = data, ntr = 1)
    prefs['simulated_transient'] = mock
    st2 = rfpipe.state.State(inmeta = meta, inprefs=prefs)

    cc = rfpipe.pipeline.pipeline_seg(st2, segment=0, devicenum = 0)

    if len(cc)>0:
        cc.prefs.clustercands = True
        cc_clustered, clusterer = rfpipe.candidates.cluster_candidates(cc, returnclusterer=True)
        if len(cc_clustered)>1 and visualize_cluster == True:
            rfpipe.candidates.visualize_clustering(cc_clustered, clusterer)

        cl_rank, cl_count = rfpipe.candidates.calc_cluster_rank(cc_clustered)
        calcinds = np.unique(np.where(cl_rank == 1)[0])
        
        #cc_clustered had no state for just one candidate. Using this generates _state.
        d_tmp = cc_clustered.state.dmarr  
        
        moc_map, label = cc_clustered.mock_map
        logging.info(f'Mock label is {label}')

        if label[0] != -2: #only for one mock, TODO: expand for any number of mocks
            det_candind = calcinds[cc_clustered.cluster[calcinds] == label][0]
            det_snr = cc_clustered.snrtot[det_candind]
            det_dm = cc_clustered.canddm[det_candind]
            det_dt = cc_clustered.canddt[det_candind]
            det_l = cc_clustered.candl[det_candind]
            det_m = cc_clustered.candm[det_candind]
        else:
            logging.info('Mock Not found.')
            det_snr = -2
            det_dm = -2
            det_dt = -2
            det_l = -2
            det_m = -2

    else:
        logging.info('Candidate was not detected.')
        det_snr = -1
        det_dm = -1
        det_dt = -1
        det_l = -1
        det_m = -1
        
    dm = mock[0][2]
    dt = mock[0][3]
    l = mock[0][5]
    m = mock[0][6]
    params.append([mock, snr, det_snr, dm, det_dm, dt, det_dt, l, det_l, m, det_m])
    
    if trial%100 == 0 and trial!=0:
        frbs = pd.DataFrame(params, columns=['mock', 'in_snr', 'det_snr', 'in_dm', 'det_dm', 'in_dt', 'det_dt', 'in_l', 
                                             'det_l', 'in_m', 'det_m'])
        if trial//100 == 1:
            frbs.to_csv('sen_all.csv', mode='a')
        else:
            frbs.to_csv('sen_all.csv', mode='a', header=False)
        params = []    
    
frbs = pd.DataFrame(params, columns=['mock', 'in_snr', 'det_snr', 'in_dm', 'det_dm', 'in_dt', 'det_dt', 'in_l', 
                                             'det_l', 'in_m', 'det_m'])
frbs.to_csv('sen_all.csv', mode='a', header=False)