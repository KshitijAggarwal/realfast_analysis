import casatasks as tasks
import casatools as tools
qa = tools.quanta()
me = tools.measures()
from astropy import coordinates, units, io
import numpy as np
import pylab as plt
import os, glob, shutil


def applycal(msfile, gaintables, gainfield=None, targetfield=None, interp=None):
    """    
    
    Apply CASA calibration using calibration tables
    
    Example Usage: 
    applycal(msfile0, gaintables, targetfield=targetfield,
         gainfield=['', '', '', '', '', '', '', ''],
         interp=['linear', 'linear', 'linear', 'linear', 'linear,linearflag', 'linear', 'linear', 'linear'])
    
    """
    if not gainfield:
        gainfield = ['' for _ in range(len(gaintables))]

    if not interp:
        interp = ['' for _ in range(len(gaintables))]

    if not targetfield:
        targetfield = '1'
        
    tasks.applycal(vis=msfile, gaintable=gaintables, gainfield=gainfield,
                   field=targetfield, interp=interp, calwt=[False])
    
    
def makeimage(msfile, field, outname='tmp.', niter=50, cell=0.5, npix=4096, 
              spw='', antenna='*', uvrange='>0lambda', weighting='natural',
              gridder='standard', robust=0.5, uvtaper=[], deconvolver='hogbom',
              conjbeams=False, wprojplanes=-1):
    """

    Tclean on the data to generate the image
    
    Example Usage:
    makeimage(msfile1, fieldnames[0], outname='IMNAME', spw='5~8', niter=100, cell=0.5, npix=4096)
    
    spw range is end inclusive
    
    """
    # To remove earlier products? 
#     %sx rm -rf $msfile?*

    wprojplanes = -1 if 'project' in gridder else 1
    tasks.tclean(vis=msfile, imagename=outname, field=field,
                 niter=niter, stokes='I', antenna=antenna, uvrange=uvrange,
                 weighting=weighting, cell=cell, imsize=[npix, npix],
                 spw=spw, gridder=gridder, robust=robust, uvtaper=uvtaper,
                 deconvolver=deconvolver, conjbeams=conjbeams, wprojplanes=wprojplanes)
    

def fitimage(image, outname='tmp.', xmin=None, xmax=None, ymin=None, ymax=None, displaywindow=300, 
             fitwindow=100, returnimfit=False, estimates=''):
    """
    
    CASA imfit on the image. Reports peak RA, DEC and errors. 
    
    """
    # load image
    ia = tools.image()
    ia.open(image)

    # summarize image
    dd = ia.summary()
    npixx,npixy,nch,npol = dd['shape']
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = npixx
    if ymin is None:
        ymin = 0
    if ymax is None:
        ymax = npixy

    print('Image shape: {0}'.format(dd['shape']))
    imvals = ia.getchunk(0, int(npixx))[xmin:xmax, ymin:ymax, 0, 0]
    peakx, peaky = np.where(imvals.max() == imvals)
    print('Peak SNR at ({0},{1}) = {2}'.format(peakx[0], peaky[0], imvals.max()/imvals.std()))
#    print('Beam shape: {0}'.format(ia.history()[1].split('\n')[10].split(':')[5]))
    
    # fit component and write residual image
    box = '{0},{1},{2},{3}'.format(xmin+peakx[0]-fitwindow//2, ymin+peaky[0]-fitwindow//2, 
                                   xmin+peakx[0]+fitwindow//2, ymin+peaky[0]+fitwindow//2)
    imfit = ia.fitcomponents(box=box, residual=outname + 'fitresid', estimates=estimates)

    # report on fit
    if imfit['converged']:
        print('{0} element(s) fit'.format(imfit['results']['nelements']))
        direction = imfit['results']['component0']['shape']['direction']

        az = direction['m0']['value']
        el = direction['m1']['value']
        az_err = direction['error']['longitude']['value']
        el_err = direction['error']['latitude']['value']
        print(direction)

        co0 = coordinates.SkyCoord(ra=np.degrees(az), dec=np.degrees(el), unit=(units.deg, units.deg))
#         peak_ra = qa.unit(qa.angle(qa.quantity(az, unitname='rad'), prec=13)[0], unitname='deg')['value']
#         peak_dec = qa.unit(qa.angle(qa.quantity(el, unitname='rad'), prec=13)[0], unitname='deg')['value']
        print('{0} +- {1}"'.format(co0.ra.degree, az_err))
        print('{0} +- {1}"'.format(co0.dec.degree, el_err))
        print('Fitpeak flux: {0} Jy'.format(imfit['results']['component0']['peak']['value']))
        print(co0.to_string('hmsdms'))
    else:
        print('fitcomponents did not converge')
    
    # load residuals
    ia = tools.image()
    ia.open(outname + 'fitresid')
    dd = ia.summary()
    npixy,npixx,nch,npol = dd['shape']
    residvals = ia.getchunk(0, int(npixx))[xmin:xmax,ymin:ymax,0,0]
    peakx_resid, peaky_resid = np.where(residvals.max() == residvals)
    print('Residual SNR at ({0},{1}) = {2}'.format(peakx_resid[0], peaky_resid[0], residvals.max()/residvals.std()))
    
    # show results
    plt.figure(figsize=(25,15))
    plt.subplot(131)
    plt.imshow(imvals.transpose() - imvals.min(), interpolation='nearest', origin='bottom')
    plt.colorbar()
    plt.subplot(132)

    plt.imshow(imvals[peakx[0]-displaywindow//2:peakx[0]+displaywindow//2, 
                     peaky[0]-displaywindow//2:peaky[0]+displaywindow//2].transpose(),
              interpolation='nearest', origin='bottom')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(residvals[peakx[0]-displaywindow//2:peakx[0]+displaywindow//2, 
                        peaky[0]-displaywindow//2:peaky[0]+displaywindow//2].transpose(),
              interpolation='nearest', origin='bottom')
    plt.colorbar()
    
    if returnimfit:
        return imfit
    else:
        return az, el, az_err, el_err


def getimage(image):
    """
    
    Loads and summarize the image, Peak SNR and position
    
    """
    # load image
    ia = tools.image()
    ia.open(image)

    # summarize image
    dd = ia.summary()
    npixx,npixy,nch,npol = dd['shape']
    print('Image shape: {0}'.format(dd['shape']))
    imvals = ia.getchunk(0, int(npixx))[:,:,0,0]
    return imvals    


def findfrb(name, msfile1, fieldname, spws, niter=100, cell=0.5, npix=2048, xmin=None, 
            xmax=None, ymin=None, ymax=None):
    """
    
    Generates the image and reports the peak position and SNR
    for a given spw range and npix. spw range is end inclusive. 
    
    """
    print(f'spw range: {spws}')
    images = glob.glob(f'{name}*')
    for temp in images:
        try:
            shutil.rmtree(temp)
        except OSError as e:  ## if failed, report it back to the user ##
            pass
            
    makeimage(msfile1, fieldname, outname=name, spw=spws, niter=niter, cell=cell, npix=npix)
    imvals = getimage(f'{name}.image')
    
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = imvals.shape[0]
    if ymin is None:
        ymin = 0
    if ymax is None:
        ymax = imvals.shape[1]

    imvals = imvals[xmin:xmax, ymin:ymax]
    
    peakx, peaky = np.where(imvals.max() == imvals)
    snr = imvals.max()/imvals.std()
    print('Peak SNR at pix ({0},{1}) = {2}'.format(peakx[0], peaky[0], snr))
#    print('Beam shape: {0}'.format(ia.history()[1].split('\n')[10].split(':')[5]))
    npixx,npixy = imvals.shape
    peakx, peaky = np.where(imvals.max() == imvals)
    peakx, peaky = peakx[0], peaky[0]
    print('------------------------')
    return snr


def image_summary(image):
    """

    Print image summary and returns data and headers

    """
    ia = tools.image()
    ia.open(image)

    # summarize image
    dd = ia.summary()

    npixx,npixy,nch,npol = dd['shape']
    print('Image shape: {0}'.format(dd['shape']))
    imvals = ia.getchunk(0, int(npixx))[:,:,0,0]

    ra_inc = dd['incr'][0]*units.radian
    dec_inc = dd['incr'][1]*units.radian

    c = ia.coordmeasures()
    direction = c['measure']['direction']
    az = direction['m0']['value']
    el = direction['m1']['value']
    co0 = coordinates.SkyCoord(ra=np.degrees(az), dec=np.degrees(el), unit=(units.deg, units.deg))
    co0_str = co0.to_string('hmsdms')
    refpix = (dd['refpix'][0], dd['refpix'][1])
    print(f'Coordinates at reference pixel {refpix} are {co0_str}')

    ra_inc = dd['incr'][0]*units.radian
    dec_inc = dd['incr'][1]*units.radian
    print(f'RA, Dec increment: {ra_inc.to(units.arcsecond).value}", {dec_inc.to(units.arcsecond).value}"')

    peakx, peaky = np.where(imvals.max() == imvals)
    peakx, peaky = peakx[0], peaky[0]
    print('Peak SNR at pix ({0},{1}) = {2}'.format(peakx, peaky, imvals.max()/imvals.std()))

    rapeak = co0.ra + (peakx - refpix[0])*ra_inc
    decpeak = co0.dec + (peaky - refpix[1])*dec_inc
    copeak = coordinates.SkyCoord(ra=rapeak,dec=decpeak).to_string('hmsdms')
    print(f'Rough coordinates of peak pixel are {copeak}')

    return imvals, dd, c


## Data summary
# import sdmpy
# file = ''
# sdm = sdmpy.SDM(file, use_xsd=False)
# scan = sdm.scan(1)
# tab = sdm['SpectralWindow']
# for row in tab:
#     print(row.getchildren())