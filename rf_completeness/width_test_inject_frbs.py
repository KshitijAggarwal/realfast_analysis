#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

from rfpipe import source, metadata, state, util, reproduce, candidates
import math, glob, os, logging, random, argparse, traceback
import numpy as np
logger = logging.getLogger(__name__)
from datetime import datetime
from astropy import time


# In[2]:


def get_gainloc(sdmname, gainpath):
    mjdi = os.path.split(sdmname)[-1].split('.')[-2]
    mjdd = os.path.split(sdmname)[-1].split('.')[-1].split('_')[0]
    mjd = float(mjdi+'.'+mjdd)
    t = time.Time(mjd, format='mjd').datetime
    gainpath = gainpath+str(t.year)+'/'+str(t.month).zfill(2)+'/'
    return gainpath


# In[3]:


# def make_one(sdms):
# snr_rec = -1
# while snr_rec < 7:  # Only save canddata when recovered SNR is more than 7


# In[4]:


# snr_rec = -1


# In[5]:


l = glob.glob('../data/17B-301_sb34374800_1_1.58060.912586516206')
sdms = l

# snr_rec = 0
# while snr_rec < 7:  # Only save canddata when recovered SNR is more than 7
sdmname = sdms[random.randint(0, len(sdms) - 1)]
logging.info(f'Let us try to inject an FRB in {sdmname}')
if sdmname[-1] == '/':
    sdmname = sdmname[:-1]

gainpath = '/hyrule/data/users/kshitij/fetchrf/gainfiles/'
preffile = '/hyrule/data/users/kshitij/fetchrf/scripts/realfast.yml'

if os.path.basename(sdmname).split('_')[0] == 'realfast':
    datasetId = '{0}'.format('_'.join(os.path.basename(sdmname).split('_')[1:-1]))
else:
    datasetId = os.path.basename(sdmname)

#     gainpath = get_gainloc(sdmname, gainpath)

gainname = datasetId + '.GN'
logging.info('Searching for the gainfile {0} in {1}'.format(gainname, gainpath))

gainfile = []
for path, dirs, files in os.walk(gainpath):
    for f in filter(lambda x: gainname in x, files):
        gainfile = os.path.join(path, gainname)
        break
try:
    assert len(gainfile)
    logging.info('Found gainfile for {0} in {1}'.format(datasetId, gainfile))
except AssertionError as err:
    logging.error('No gainfile found for {0} in {1}'.format(datasetId, gainfile))        
    raise err

scans = list(util.getsdm(sdmname).scans())
intents = [scan.intents for scan in scans]
logger.info("Found {0} scans of intents {1} in {2}"
            .format(len(scans), intents, sdmname))
intent = 'TARGET'
scannums = [int(scan.idx) for scan in scans
            if scan.bdf.exists and any([intent in scint for scint in scan.intents])]
logger.info("Found {0} scans with intent {1} of total {2} scans".format(len(scannums), intent, len(scans)))
scannum = scannums[random.randint(0, len(scannums) - 1)]

band = metadata.sdmband(sdmfile=sdmname, sdmscan=scannum)

try:
    st = state.State(sdmfile=sdmname, sdmscan=scannum, preffile=preffile, name='NRAOdefault'+band
                 , showsummary=False)
except ValueError:
    prefs = {'spw':None}
    st = state.State(sdmfile=sdmname, sdmscan=scannum, preffile=preffile, name='NRAOdefault'+band
                 , showsummary=False, inprefs=prefs)

st.prefs.gainfile = gainfile
st.prefs.workdir = '/hyrule/data/users/kshitij/fetchrf/sim_frbs/' #fetch_data_dir+datasetId
logging.info('Working directory set to {0}'.format(st.prefs.workdir))

nseg = st.nsegment
# Random segment choice 
segment = random.randint(0,nseg-1)

data = source.read_segment(st,segment)

# if not any(data): # Should fix: Error: axis 3 is out of bounds for array of dimension 1 occured
#     continue

# Random location of candidate in the radio image
l = math.radians(random.uniform(-st.fieldsize_deg/2, st.fieldsize_deg/2))
m = math.radians(random.uniform(-st.fieldsize_deg/2, st.fieldsize_deg/2))

# Random SNR choice
snr = 0
while snr < 8:
    snr = np.random.lognormal(2.5,1)
#     snr  = random.uniform(10,100)


# In[6]:


data_orig = source.read_segment(st,segment)
if not np.any(data_orig): # fix for -> Error: axis 3 is out of bounds for array of dimension 1 occured
    logger.info('Data is all zeros or empty. Trying a new injection.')
#         continue


# In[7]:


injected_loc = []
shortcut_snr = []
shortcut_loc = []
pipeline_snr = []
pipeline_loc = []


# In[8]:


# for dtind in [2,3]:
for temp in range(100):
    i = 150
    dmind = 5
    snr = 60
    st.clearcache()
    st.prefs.maxdm = 150
    st.dmarr, st.dmarr[5]
    # Random choice of segment, i, dm (0-maxdm), dt (0-maxdt) with input l, m, snr chosen randomly as above
#     mock = util.make_transient_params(st, snr = snr, segment=segment, data = data, ntr = 1, dmind = None, 
#                                       dtind = 3, lm = (l,m))
    mock = util.make_transient_params(st, snr = snr, segment=segment, i = i, data = data_orig, ntr = 1, 
                                      dmind = dmind, lm = (l,m))

    st.prefs.simulated_transient = mock
    st.prefs.savecanddata = False

    injected_loc.append(mock[0])

    # From candidates.mock_map , chose the closest candloc from mock props
    (segment, integration, dm, dt, amp, l0, m0) = mock[0]
    dmind0 = np.abs((np.array(st.dmarr)-dm)).argmin()
    dtind0 = np.abs((np.array(st.dtarr)*st.inttime-dt)).argmin()
    integration0 = integration//2**dtind0

    integrations = [i for i in [integration0-1, integration0, integration0+1] if i >= 0 and i < st.nints]
    dms = [dm for dm in [dmind0-1, dmind0, dmind0+1] if dm >=0 and dm < len(st.dmarr)]
    mocklocs = []
    for i in integrations:
        for dm in dms:
            mocklocs.append((segment, i, dm, dtind0, 0))

#     mockloc = (segment, integration0, dmind0, dtind0, 0)
    logger.info(f'Injecting mock transient with parameters:{mock[0]}')
    data = source.read_segment(st, segment)
    data = source.data_prep(st, segment, data)
    if not np.any(data): # Should fix: TypeError: can't convert complex to float
        logger.info('Data is all zeros or empty. Trying a new injection.')
#         continue

    logger.info(f'Trying to find the injected transient at mocklocs: {mocklocs}')
    cds = []
    snrs = []
    for mockloc in mocklocs:
        data_corr = reproduce.pipeline_datacorrect(st, mockloc, data_prep=data)
        cd_t = reproduce.pipeline_canddata(st, mockloc, data_corr)
        cds.append(cd_t)
        snrs.append(cd_t.snr1)

    (m_segment, m_i, m_dm, m_dt, m_amp, m_l, m_m) = mock[0]        

    cd = cds[np.argmax(snrs)]

    shortcut_snr.append(cd.snr1)
    shortcut_loc.append(cd.loc)

    snr_rec = cd.snr1
    logger.info(f'Injected SNR was {snr}, max recovered snr is {snr_rec}.')
    if snr_rec > 7:
        logger.info('Recovered SNR is greater than 7. Yaaaay for successful injection!!')

    from rfpipe import pipeline
    st.prefs.clustercands = False
    st.prefs.savecandcollection = False
    st.prefs.returncanddata = False
    st.prefs.saveplots = False
    cc_unclustered = pipeline.pipeline_seg(st=st, segment=segment, devicenum=0)

    pipeline_loc.append(cc_unclustered.locs[cc_unclustered.array['snr1'].argmax()])
    pipeline_snr.append(cc_unclustered.array['snr1'].max())    


# In[13]:

d = [injected_loc, pipeline_snr, shortcut_snr, pipeline_loc, shortcut_loc]


# In[47]:


import pickle


# In[48]:


with open('outfile', 'wb') as fp:
    pickle.dump(d, fp)



import pylab as plt

# In[26]:


equal_line = np.linspace(min(pipeline_snr), max(pipeline_snr), 100)
widths = [loc[3] for loc in injected_loc]


# In[40]:


plt.scatter(x=shortcut_snr, y=pipeline_snr)
plt.plot(equal_line,equal_line)
plt.xlabel('Shortcut SNR')
plt.ylabel('Pipeline SNR')
plt.grid()
plt.savefig('shortcut_vs_pipeline_snr.png')


# In[41]:


plt.scatter(x=widths, y=shortcut_snr, label='Recovered SNR')
plt.hlines(y=60, xmin=0, xmax=0.03, linestyles='dashed', label='Injected SNR')
plt.ylabel('SNR')
plt.xlabel('Injected Width (ms)')
plt.legend()
plt.xlim([0, 0.03])
plt.grid()
plt.savefig('Recovered_SNR_vs_width.png')


# In[45]:

















