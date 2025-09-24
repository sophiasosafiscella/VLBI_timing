This is a GitHub repository for the source code used in the publication titled "The NANOGrav 15-Year Data Set: Improved Timing Precision With VLBI Astrometric Priors".

Our software scripts and the input data files are available from our GitHub repository (https://github.com/sophiasosafiscella/VLBI_sporc). Our Python code requires the `astropy`, `PINT`, and `uncertainties` packages. The input `.par` and `.tim` pulsar timing data files are available from https://zenodo.org/record/7967584. The VLBI positions are obtained from the literature and stored in `msp_vlbi.csv`. The calibration source positions for the millisecond pulsars are stored in `cal.csv`.

The scripts must be executed in the following order:

1. `1_get_timing_data.py`: Get the full NANOGrav timing astrometry while also updating the epochs to match those from VLBI
2. `2_calibrate_VLBI_data.py`: Calibrate the VLBI astrometric positions to the RFC.
3. `3_find_frame_tie.py`: Find the frame tie between the reference frame used in timing and the reference frame defined by the RFC.
4. `4_apply_frame_tie.py`: Apply the frame tie to convert the VLBI astrometric values to the reference frame used in timing.
5. `5_find_overlaps.py`: Find the overlaps between a 3-sigma range around the timing- and VLBI-derived astrometric values to create the different trial astrometric solutions.
6. `6_job_array_bundle.sh`: This is a bash script that parallelizes `calculate_posterior.py`, running it simultaneously for several different trial astrometric solutions.
7. `7_consolidate_posteriors.py`: Put all the posteriors (and corresponding trial astrometric solutions) into a single file.

`VLBI_utils.py` contains a series of helper functions that are invoked by the other scripts.

The folder `plotting_scripts` contains a series of scripts to recreate the plots presented in the paper.
