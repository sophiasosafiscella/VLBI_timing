# Get only the positions from NANOGrav timing astrometry while also updating the epochs to match those from VLBI

import contextlib
import glob

import numpy as np
import pandas as pd
import pint.fitter
from pint.models import get_model
from pint.toa import get_TOAs

PSR_list = pd.read_csv('./data/NG_frame_tie/mspsr_list.txt', sep=',', header=None).to_numpy()[0]
n_psrs = len(PSR_list)

new_epochs = pd.read_csv('./data/NG_frame_tie/NG_msp_vlbi.csv', sep=',', comment='#')['epoch_v'].to_numpy()

RA_list, RA_err_list, DEC_list, DEC_err_list, EPHEM_list = [np.empty(n_psrs, dtype=object) for _ in range(5)]
POSEPOCH_list = np.empty(n_psrs, dtype=float)
corr_list = np.empty(n_psrs, dtype=float)

for k, psr in enumerate(PSR_list):

    PSR_name = psr + "_PINT"
    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    eq = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    eq.change_posepoch(new_epochs[k])

    RA_list[k] = eq.RAJ.quantity
    RA_err_list[k] = eq.RAJ.uncertainty
    DEC_list[k] = eq.DECJ.quantity
    DEC_err_list[k] = eq.DECJ.uncertainty
    POSEPOCH_list[k] = eq.POSEPOCH.value
    EPHEM_list[k] = eq.EPHEM.value

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


data = pd.DataFrame({'ra_t': RA_list, 'ra_te': RA_err_list, 'dec_t': DEC_list, 'dec_te': DEC_err_list,
        'epoch_t': POSEPOCH_list, 'ephem': EPHEM_list, 'cor': corr_list}, index=PSR_list)

data.to_csv("./data/NG_frame_tie/NG_msp_timing.csv")

