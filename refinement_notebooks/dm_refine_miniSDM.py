#!/usr/bin/env python
# coding: utf-8

from rfpipe import pipeline, candidates, state
# Updated for rfpipe version 1.4.1
import numpy as np 
import pylab as plt
import matplotlib
import logging
import glob
logger = logging.getLogger('rfpipe')
from refine_utils import cd_refine    
import argparse
import os


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

def refine(miniSDMlist, workdir, preffileloc, gainpath = '/home/mchammer/evladata/telcal/', 
           search_sigma = 7, refine = True, classify = True, dm = 350, dm_frac=0.2, dm_steps = 100, devicenum = '0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = devicenum
    # Searching for gainfile
    gainfile = []
    for sdm in miniSDMlist:
        sdmname = sdm.split('/')[-1]
        datasetId = '{0}'.format('_'.join(sdmname.split('_')[1:-1]))
        # # set the paths to the gainfile
        gainname = datasetId + '.GN'
        logging.info('Searching for the gainfile {0} in {1}'.format(gainname, gainpath))
        for path, dirs, files in os.walk(gainpath):
            for f in filter(lambda x: gainname in x, files):
                gainfile.append(os.path.join(path, gainname))
                break

    # Searching all miniSDMs
    for index, sdm in enumerate(miniSDMlist):
        prefs = {'saveplots': True, 'savenoise': False, 'savesols': False, 'savecandcollection': True, 
                 'savecanddata': True, 'workdir': workdir, 'gainfile':gainfile[index], 
                 'sigma_image1': search_sigma, 'dmarr': list(np.linspace(dm-dm_frac*dm, dm+dm_frac*dm, dm_steps))}

        st = state.State(sdmfile=sdm, sdmscan=1, inprefs=prefs, preffile = preffileloc, name='NRAOdefaultL')
        cc = pipeline.pipeline_scan(st)

    # # Classify and generate refinement plots
    
    # Classify the generated pickles using FETCH and generate refinement plots
    for miniSDM in miniSDMlist:
        sdmname = miniSDM.split('/')[-1]
        for pkl in glob.glob(st.prefs.workdir+'/'+'cands_*'+sdmname.split('/')[0]+'*.pkl'):
            if classify or refine:
                logging.info('Refining and classifying pkl: {0}'.format(pkl))
                ccs = list(candidates.iter_cands(pkl, select='candcollection'))
                for cc in ccs:
                    cds = cc.canddata
                    if cds:
                        for cd in cds:
                            if classify:
                                payload = candidates.cd_to_fetch(cd, classify=True, save_png=True, show=True, mode = 'GPU'
                                                                 , outdir = workdir, devicenum='0')
                                logging.info('FETCH FRB Probability of the candidate {0} is {1}'.format(cd.candid, payload))
                            if refine:
                                logging.info('Generating Refinement plots')
                                cd_refine(cd, save=True, outdir = workdir)
                    else:
                        logging.info('No candidate was found in cc: {0}'.format(cc))
                        
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine DM refinement of rfpipe candidates")
    parser.add_argument('-sdm', '--sdm_path', help='Path of SDMs', required=True, type=str)
    parser.add_argument('-o', '--workdir', help='Working Directory', required=True, type=str)
    parser.add_argument('-g', '--gainpath', help='Path to gainfile directories', required=False, 
                        type=str, default='/home/mchammer/evladata/telcal/')
    parser.add_argument('-p', '--preffileloc', help='Path to preferences file', required=False, 
                        type=str, default='/lustre/aoc/projects/fasttransients/staging/realfast.yml')
    parser.add_argument('-dm', '--DM', help='DM of the candidate', required=True, type=int)
    parser.add_argument('-dmf', '--DM_fraction', help='Fraction of DM around central DM to search',
                        required=False, type=float, default=0.3)
    parser.add_argument('-dms', '--DM_steps', help='Number of DM steps', required=False, type=int, default=200)
    parser.add_argument('-s', '--search_sigma', help='Sigma of the image for search', required=False, type=float, default=7)
    parser.add_argument('-r', '--refinement_plots', help='Generate and Save refinement plots', action='store_true')
    parser.add_argument('-c', '--classify', help='Classify the candidates using FETCH', action='store_true')
    parser.add_argument('-d', '--devicenum', help='Enter the GPU number to run pipeline', type=str, default=0, required=False)
    args = parser.parse_args()
    
    refine(miniSDMlist=glob.glob(args.sdm_path), workdir=args.workdir, preffileloc=args.preffileloc, 
           gainpath=args.gainpath, search_sigma = args.search_sigma, refine = args.refinement_plots,
           classify = args.classify, dm = args.DM, dm_frac=args.DM_fraction, dm_steps = args.DM_steps, devicenum=args.devicenum)
