from rfpipe import candidates

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy import coordinates as coord
import logging
logger = logging.getLogger()
logger = logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s -'
                                                        ' %(message)s')
                                                        
def query_psrcat(cd, radius, psrcat_loc):
    """ Takes canddata of the candidate, radius in arcminnutes (in astropy.coordinates)  and 
    location of psrcat (csv) as input and returns a dataframe of the pulsars found within
    the specified radius of the candidate. 
    """

    l1 = cd.l1
    m1 = cd.m1
    pt_ra, pt_dec = cd.state.metadata.radec
    src_ra_deg, src_dec_deg = candidates.source_location(pt_ra, pt_dec, l1, m1)
    center = SkyCoord(src_ra_deg + ' ' + src_dec_deg, unit=(u.hourangle, u.deg), frame='icrs')
    
    radius = coord.Angle(radius)    
    r_deg = radius.to('deg').value

    ra = center.ra.deg
    dec = center.dec.deg
    
    ra_min = ra - r_deg
    ra_max = ra + r_deg
    dec_min = dec - r_deg
    dec_max = dec + r_deg
    
    psrcat_df = pd.read_csv(psrcat_loc, skiprows=2, usecols = [1, 3, 6, 9, 10],  \
                        names = ['Name', 'DM', 'P0', 'RAJD', 'DECJD'])
    
    psrcat_region_mask = (psrcat_df['RAJD'] > ra_min) & (psrcat_df['RAJD'] < ra_max) & \
                         (psrcat_df['DECJD'] > dec_min) & (psrcat_df['DECJD'] < dec_max)
    
    rt_psrcat = psrcat_df[psrcat_region_mask]
    logging.info(f'Found {len(rt_psrcat)} pulsars in PSRCAT within {radius} of (ra = {ra:.4f}deg, dec = {dec:.4f}deg')
    return rt_psrcat


def query_frbcat(cd, radius, frbcat_loc):
    """ Takes canddata of the candidate, radius in arcminnutes (in astropy.coordinates)  and 
    location of frbcat (csv) as input and returns a dataframe of the frbs found within
    the specified radius of the candidate. 
    """

    l1 = cd.l1
    m1 = cd.m1
    st = cd.state
    pt_ra, pt_dec = st.metadata.radec
    src_ra_deg, src_dec_deg = candidates.source_location(pt_ra, pt_dec, l1, m1)
    center = SkyCoord(src_ra_deg + ' ' + src_dec_deg, unit=(u.hourangle, u.deg), frame='icrs')
    
    radius = coord.Angle(radius)    
    r_deg = radius.to('deg').value

    ra = center.ra.deg
    dec = center.dec.deg
    
    ra_min = ra - r_deg
    ra_max = ra + r_deg
    dec_min = dec - r_deg
    dec_max = dec + r_deg
    
    frbcat_df = pd.read_csv(frbcat_loc)    
    ra_deg = []
    dec_deg = []
    for idx, row in frbcat_df.iterrows():
        center = SkyCoord(l=row['rop_gl']*u.deg, b=row['rop_gb']*u.deg, frame='galactic')
        ra_deg.append(center.icrs.ra.deg)
        dec_deg.append(center.icrs.dec.deg)
        
    frbcat_df['RAJD'] = ra_deg
    frbcat_df['DECJD'] = dec_deg    
    
    frbcat_region_mask = (frbcat_df['RAJD'] > ra_min) & (frbcat_df['RAJD'] < ra_max) & \
                         (frbcat_df['DECJD'] > dec_min) & (frbcat_df['DECJD'] < dec_max)
    rt_frbcat = frbcat_df[frbcat_region_mask]
    
    logging.info(f'Found {len(rt_frbcat)} pulsars in FRBCAT within {radius} of (ra = {ra:.4f}deg, dec = {dec:.4f}deg')
    return rt_frbcat


