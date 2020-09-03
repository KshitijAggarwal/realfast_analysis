from rfpipe import state, source, calibration, flagging, candidates
from rfpipe.search import dedisperseresample
from rfpipe.util import calc_delay, get_uvw_segment, phase_shift
import numpy as np


def prepare_data(sdmfile, gainfile, delta_l, delta_m, segment=0, dm=0, dt=1, spws=None):
    """
    
    Applies Calibration, flagging, dedispersion and other data preparation steps
    from rfpipe. Then phaseshifts the data to the location of the candidate. 
    
    """
    st = state.State(sdmfile=sdmfile,
                        sdmscan=1, inprefs= {'gainfile': gainfile, 
                                            'workdir': '.', 'maxdm':0, 'flaglist': []}, showsummary=False)
    if spws:
        st.prefs.spw = spws
            
    data = source.read_segment(st, segment)
    
    takepol = [st.metadata.pols_orig.index(pol) for pol in st.pols]
    takebls = [st.metadata.blarr_orig.tolist().index(list(bl)) for bl in st.blarr]
    datap = np.require(data, requirements='W').take(takepol, axis=3).take(st.chans, axis=2).take(takebls, axis=1)
    datap = source.prep_standard(st, segment, datap)
    datap = calibration.apply_telcal(st, datap)
    datap = flagging.flag_data(st, datap)

    delay = calc_delay(st.freq, st.freq.max(), dm, st.inttime)
    data_dmdt = dedisperseresample(datap, delay, dt)
    
    print(f'shape of data_dmdt is {data_dmdt.shape}')
    
    uvw = get_uvw_segment(st, segment)
    phase_shift(data_dmdt, uvw=uvw, dl=delta_l, dm=delta_m)
    
    dataret = data_dmdt
    return dataret, st


def get_maxsnr_cd(pkl):
    """
    
    Return the canddata of the candidate with maximum snr from an
    input pkl file.
    
    """
    ccs = list(candidates.iter_cands(pkl, select='candcollection'))
    assert len(ccs) == 1
    cc = ccs[0]
    cds = list(candidates.iter_cands(pkl, select='canddata'))
    assert len(cds) == 1
    cds = cds[0]
    cd = cds[np.argmax(cc.snrtot)]
    return cd


def calc_cand_integration(cd):
    """
    
    Calculates the integration of the candidate using the canddata.
    The integration is calcualted with respect to the start of the SDM, 
    referenced to the max frequency of the band. 
    
    """    
    segment, candint, dmind, dtind, beamnum = cd.loc
    st = cd.state
    dm = st.dmarr[dmind]
    name = cd.state.metadata.filename.split('/')[-1] + '/'
    integration = candint
    dt = cd.state.dtarr[dtind]
    inttime = cd.state.inttime
    
    orig_max_freq = np.max(cd.state.metadata.freq_orig)
    obs_max_freq = np.max(cd.state.freq)
    delay_ints = calc_delay(freq=[obs_max_freq], freqref=orig_max_freq, dm=dm, inttime=cd.state.inttime)[0]
    
    i = segment*cd.state.readints + integration*cd.state.dtarr[dtind] - delay_ints - 1
    return i, name, dm