"""
    For the calibration source of each pulsar in the dataset:
    1) Convert the errors in RA and DEC to milliarcseconds.
    2) Look up the source in the Radio Fundamentals Catalog (RFC)
    3) Create a table containing the RA and DEC of each calibration source, both in the original catalog and in the RFC

    Input data
    ----------
    - Table containing the RFC (cal_data.csv)
    - Table containing each pulsar, its calibrator, and the calibrator's RA and DEC if it is not originally in the RFC
    (rfc_2025c_cat.txt)

    Returns
    -------
    Table containing:
    - each pulsar,
    - its calibrator,
    - the calibrator's RA and DEC in both the original catalog and in the RFC
    - the errors in RA and DEC in milliarcseconds
"""

import numpy as np
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import pandas as pd
from VLBI_utils import handle_error
import sys

# Define column positions and names based on the file's header description
col_specs = [
    (0, 3), (4, 14), (16, 24), (26, 28), (29, 31), (32, 41), (42, 43),
    (43, 45), (46, 48), (49, 57), (59, 65), (66, 72)
]
names = ['RFC',
    'Name', 'Comnam', 'RAh', 'RAm', 'RAs', 'DE-',
    'DEd', 'DEm', 'DEs', 'eRA', 'eDE'
]

# Read input data
rfc = pd.read_fwf('./data/rfc_2025c_cat.txt', colspecs=col_specs, names=names, index_col='Name', comment='#')
cal_data = pd.read_table('./data/cal_data.csv', header=0, index_col=0, sep=',', comment='#')
cal_sources_list = cal_data['cal_source']
n_psrs = len(cal_sources_list)

# Create arrays to store output data
ra_og, dec_og, ra_err_og, dec_err_og = [np.empty(n_psrs, dtype=object) for _ in range(4)]
ra_rfc, dec_rfc, ra_err_rfc, dec_err_rfc = [np.empty(n_psrs, dtype=object) for _ in range(4)]

# Iterate over the pulsars
for index, (psr_name, row) in enumerate(cal_data.iterrows()):

    is_rfc_original = (row['og_cat'] == 'RFC')

    if not is_rfc_original:
        # Read the RA and DEC in the original catalog and convert their errors to milliarcseconds
        og_pos = SkyCoord(ra=row['og_cal_ra'], dec=row['og_cal_dec'], unit=(u.hourangle, u.deg))
        ra_og[index], dec_og[index] = og_pos.to_string('hmsdms').split()

        # Handle the various possible input units for the error in RA and DEC
        ra_err_og[index] = handle_error(row['og_cal_rae'], og_pos.dec.degree)
        dec_err_og[index] = handle_error(row['og_cal_dece'])

    # Now, always look up the calibrator in the RFC catalog
    try:
        cal_source_data = rfc.loc[row['cal_source']]  # Data of the calibration source
        ra_str = f"{cal_source_data['RAh']}h{cal_source_data['RAm']}m{cal_source_data['RAs']}s"
        dec_str = f"{cal_source_data['DE-']}{cal_source_data['DEd']}d{cal_source_data['DEm']}m{cal_source_data['DEs']}s"

        rfc_pos = SkyCoord(ra=ra_str, dec=dec_str)
        ra_rfc[index], dec_rfc[index] = rfc_pos.to_string("hmsdms").split()

        # RFC errors are already in mas
        ra_err_rfc[index] = cal_source_data['eRA'] * u.mas
        dec_err_rfc[index] = cal_source_data['eDE'] * u.mas

    except KeyError:
        # Handle cases where the calibrator isn't found in the RFC
        print(f"Warning: Calibrator '{row['cal_source']}' not found in RFC catalog.", file=sys.stderr)
        ra_rfc[index], dec_rfc[index], ra_err_rfc[index], dec_err_rfc[index] = [np.nan] * 4

# Output the results
results = pd.DataFrame({
    'cal_source': cal_sources_list,
    'og_cat': cal_data['og_cat'],
    'og_cal_ra': ra_og,
    'og_cal_rae': ra_err_og,
    'og_cal_dec': dec_og,
    'og_cal_dece': dec_err_og,
    'RFC_cal_ra': ra_rfc,
    'RFC_cal_rae': ra_err_rfc,
    'RFC_cal_dec': dec_rfc,
    'RFC_cal_dece': dec_err_rfc
})

results.to_csv('./data/full_cal_data.csv', header=True)