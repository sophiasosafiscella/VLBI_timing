# Get the full NANOGrav timing astrometry while also updating the epochs to match those from VLBI

import numpy as np
import pandas as pd
from pint.models import get_model
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import Angle
from math import sqrt
import glob
import sys

VLBI_data = pd.read_csv('./data/msp_vlbi.csv', sep=',', comment='#', index_col=0)
PSR_list = VLBI_data.index.tolist()
n_psr = len(PSR_list)
new_epochs = VLBI_data['epoch_v'].to_numpy()

RA_list, RA_err_list, DEC_list, DEC_err_list, EPHEM_list, equinox_list = [np.empty(n_psr, dtype=object) for _ in range(6)]
PMRA_list, PMRA_err_list, PMDEC_list, PMDEC_err_list, PX_list, PX_err_list = [np.empty(n_psr, dtype=float) for _ in range(6)]
POSEPOCH_list = np.empty(n_psr, dtype=float)

for k, psr in enumerate(PSR_list):

    PSR_name = psr + "_PINT"
    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)                                # Ecliptical coordiantes
    eq = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)  # Equatorial coordinates

    # Compute time difference
    time_diff = Time(new_epochs[k], format='mjd') - Time(eq.POSEPOCH.value, format='mjd')  # This is a TimeDelta object
    timespan_in_years = time_diff.to_value('year') * u.year

    # Update the epoch to match that of VLBI
    eq.change_posepoch(new_epochs[k])

    # Update positions
    RA_list[k] = eq.RAJ.quantity
    DEC_list[k] = eq.DECJ.quantity

    # Update uncertainties, taking into account that when we apply proper motion,
    # the components of proper motion also have errors that will be propagated into the positions
    RA_err_list[k] = Angle(sqrt(Angle(eq.RAJ.uncertainty, unit=u.hourangle).rad**2 + Angle(eq.PMRA.uncertainty * timespan_in_years, unit=u.mas).rad**2), unit=u.rad).to_string(unit=u.hourangle)
    DEC_err_list[k] = Angle(sqrt(Angle(eq.DECJ.uncertainty, unit=u.degree).rad**2 + Angle(eq.PMDEC.uncertainty * timespan_in_years, unit=u.mas).rad**2), unit=u.rad).to_string(unit=u.degree)

    PMRA_list[k] = eq.PMRA.value
    PMRA_err_list[k] = eq.PMRA.uncertainty.value
    PMDEC_list[k] = eq.PMDEC.value
    PMDEC_err_list[k] = eq.PMDEC.uncertainty.value

    PX_list[k] = eq.PX.value
    PX_err_list[k] = eq.PX.uncertainty.value

    POSEPOCH_list[k] = eq.POSEPOCH.value
    EPHEM_list[k] = eq.EPHEM.value
    equinox_list[k] = "J2000.0"


data = pd.DataFrame({'epoch_t': POSEPOCH_list, 'ephem': EPHEM_list, 'equinox': equinox_list,
                     'ra_t': RA_list, 'ra_te': RA_err_list, 'dec_t': DEC_list, 'dec_te': DEC_err_list,
                     'pmra_t': PMRA_list, 'pmra_te': PMRA_err_list, 'pmdec_t': PMDEC_list, 'pmdec_te': PMDEC_err_list,
                     'px_t': PX_list, 'px_te': PX_err_list}, index=PSR_list)

data.to_csv("./data/timing_astrometric_data_updated.csv")

