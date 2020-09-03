Scripts to do refinement of realfast candidates and improve the localisation using CASA calibration + fitting. 

* `refine_candidate.ipynb`: Refine candidate DM and other props 
using [refine_sdm](https://github.com/realfastvla/rfpipe/blob/2bb6ab2237b61ee46e42d36e38a9c978d3d9a6ba/rfpipe/reproduce.py#L245). Generate a pickle file
with canddata. 

* `ddcut_signal_integration.ipynb` : Use the pickle file generated in prev step and figure out the integration(s) to cut out for imaging. 
Uses [sdmpy](https://github.com/demorest/sdmpy) to generate a dedispersed SDM containing the candidate signal. Then convert the SDM to MS file using `asdm2MS`. 

* `casa_imaging.ipynb` : Read the MS file created in prev step, figure out the optimal spw range, and image using CASA. Uses `imfit` to fit for a position
and reports the peak position, flux and fit errors. 

