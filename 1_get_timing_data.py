#----------------------------------------------------------------------------------------------------------------------
# 1) Get the full NANOGrav timing astrometry while also updating the epochs to match those from VLBI
#----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import contextlib
import pint.fitter
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord
from VLBI_utils import ra_error_to_mas
from math import sqrt
import glob

VLBI_data = pd.read_csv('./data/msp_vlbi.csv', sep=',', comment='#', index_col=0)
PSR_list = VLBI_data.index.tolist()
n_psr = len(PSR_list)
new_epochs = VLBI_data['epoch_v'].to_numpy()

RA_list, RA_err_list, DEC_list, DEC_err_list, EPHEM_list, equinox_list = [np.empty(n_psr, dtype=object) for _ in range(6)]
PMRA_list, PMRA_err_list, PMDEC_list, PMDEC_err_list, PX_list, PX_err_list = [np.empty(n_psr, dtype=float) for _ in range(6)]
POSEPOCH_list = np.empty(n_psr, dtype=float)
corr_list = np.full(n_psr, np.nan)

for k, psr in enumerate(PSR_list):

    print(f"Processing PSR {psr}", flush=True)
    PSR_name = psr + "_PINT"
    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)                                # Ecliptical coordiantes
    eq = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)  # Equatorial coordinates

    # Compute time difference
    timespan_in_years = (Time(new_epochs[k], format='mjd') - Time(eq.POSEPOCH.value, format='mjd')).to_value('year')

    # Update the epoch to match that of VLBI
    eq.change_posepoch(new_epochs[k])

    # Update positions
    pos = SkyCoord(ra=eq.RAJ.quantity, dec=eq.DECJ.quantity, unit=(u.hourangle, u.deg))
    RA_list[k] = pos.ra.to_string(unit=u.hourangle)
    DEC_list[k] = pos.dec.to_string(unit=u.deg)

    # Update uncertainties, taking into account that when we apply proper motion,
    # the components of proper motion also have errors that will be propagated into the positions

    RA_err_mas = ra_error_to_mas(eq.RAJ.uncertainty.hms[-1], pos.dec.degree)
    RA_err_total_mas = sqrt(RA_err_mas ** 2 +(eq.PMRA.uncertainty.value * timespan_in_years) ** 2)
    RA_err_list[k] = RA_err_total_mas * u.mas

    DEC_err_total_mas = sqrt((eq.DECJ.uncertainty.to(u.mas).value) ** 2 + (eq.PMDEC.uncertainty.value * timespan_in_years) ** 2)
    DEC_err_list[k] = DEC_err_total_mas * u.mas

    PMRA_list[k] = eq.PMRA.value
    PMRA_err_list[k] = eq.PMRA.uncertainty.value
    PMDEC_list[k] = eq.PMDEC.value
    PMDEC_err_list[k] = eq.PMDEC.uncertainty.value

    PX_list[k] = eq.PX.value
    PX_err_list[k] = eq.PX.uncertainty.value

    POSEPOCH_list[k] = eq.POSEPOCH.value
    EPHEM_list[k] = eq.EPHEM.value
    equinox_list[k] = "J2000.0"

    # Calculate the correlations

    # Import TOAs
    toas = get_TOAs(timfile, planets=True, ephem=eq.EPHEM.value)

    # For the correlation matrix:
    # https://nanograv-pint.readthedocs.io/en/latest/examples/covariance.html#Extracting-the-parameter-covariance-matrix
    fitter = pint.fitter.Fitter.auto(toas, eq)
    with contextlib.suppress(pint.fitter.MaxiterReached):
        fitter.fit_toas(maxiter=0)

    pint_correlations = fitter.parameter_correlation_matrix

    params = fitter.model.free_params
    corr_labels = [label for label, _ in pint_correlations.labels[0]]
    ix = [corr_labels.index(p) for p in params]

    raw_correlation = pint_correlations.matrix
    assert np.allclose(raw_correlation, raw_correlation.T)
    raw_correlation = (raw_correlation + raw_correlation.T) / 2
    # extract rows in the right order then columns in the right order
    correlation = (raw_correlation[ix, :])[:, ix]

    assert correlation.shape == (len(params), len(params))

    for i, p1 in enumerate(params):
        assert p1 in corr_labels
        for j, p2 in enumerate(params[: i + 1]):
            assert (
                    correlation[i, j]
                    == raw_correlation[corr_labels.index(p1), corr_labels.index(p2)]
            )
            assert correlation[i, j] == correlation[j, i]

    correlation_list = [
        (p1, p2, correlation[i, j])
        for i, p1 in enumerate(params)
        for j, p2 in enumerate(params[:i])
    ]

    for p1, p2, c in correlation_list:
        if p1 == "DECJ" and p2 == "RAJ":
            #            print(f"{p1:10s} {p2:10s} {c:+.15f}")
            corr_list[k] = c


data = pd.DataFrame({'epoch_t': POSEPOCH_list, 'ephem': EPHEM_list, 'equinox': equinox_list,
                     'ra_t': RA_list, 'ra_te': RA_err_list, 'dec_t': DEC_list, 'dec_te': DEC_err_list,
                     'pmra_t': PMRA_list, 'pmra_te': PMRA_err_list, 'pmdec_t': PMDEC_list, 'pmdec_te': PMDEC_err_list,
                     'px_t': PX_list, 'px_te': PX_err_list, 'cor': corr_list}, index=PSR_list)

data.to_csv("./data/timing_astrometric_data_updated.csv")

