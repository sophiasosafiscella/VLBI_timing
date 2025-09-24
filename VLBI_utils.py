import astropy.units as u
import numpy as np
import pandas as pd
import scipy
import astropy
from astropy.coordinates import Angle, spherical_to_cartesian, cartesian_to_spherical, ICRS, SkyCoord
from astropy.time import Time
import pint
from pandas.core.frame import pandas
from pint.models.timing_model import TimingModel
from pypulse.utils import weighted_moments
from scipy.stats import norm, skewnorm
from uncertainties import ufloat, umath
from math import log as ln

import sys

def umath_spherical_to_cartesian(spherical):
    """Converts spherical coordinates (rho, ra, dec) to Cartesian coordinates (x, y, z),"""
    # Note that the input angles should be in latitude/longitude or elevation/azimuthal form.
    # I.e., the origin is along the equator rather than at the north poles

    x = umath.cos(spherical["dec"]) * umath.cos(spherical["ra"])
    y = umath.cos(spherical["dec"]) * umath.sin(spherical["ra"])
    z = umath.sin(spherical["dec"])

    return dict(x=x, y=y, z=z)

def umath_cartesian_to_spherical(cartesian):
    """Converts Cartesian coordinates (x, y, z) to spherical coordinates (ra, dec)."""

    # Use umath's atan2 and acos, which propagate uncertainties automatically
    ra = umath.atan2(cartesian["y"], cartesian["x"])  # Safely computes RA with uncertainty
    dec = umath.asin(cartesian["z"])                  # Safely computes DEC with uncertainty

    return dict(ra=ra, dec=dec)

def parSkewNormal(x0, uL, uR, pX=0.5, pL=0.025, pR=0.975, wX=1, wL=1, wR=1):
    ## INPUTS
    ## x  = Measured value   : x is the 100*pX percentile
    ## VLBI_uL = Left uncertainty : x - VLBI_uL is the 100*pL percentile
    ## VLBI_uR = Right uncertainty: x + VLBI_uR is the 100*pR percentile
    ## wX, wL, wR = Weights for the errors made when attempting to
    ## reproduce x, x-VLBI_uL, and x+VLBI_uR as percentiles of a skew-normal
    ## distribution
    ## OUTPUT
    ## Vector with the values of xi, omega, and alpha for the best
    ## fitting skew-normal distribution

    # xi : vector of location parameters.
    # omega : vector of scale parameters; must be positive.
    # alpha : vector of slant parameter(s); +/- Inf is allowed. For psn, it must be of length 1 if engine="T.Owen". For qsn, it must be of length 1.

    if any(np.array([wX, wL, wR]) < 0):
        raise ValueError("ERROR in parSkewNormal: Weights wL, wX, and wR must all be positive")
    if not ((pL < pX) and (pX < pR)):
        raise ValueError("ERROR in parSkewNormal: Probabilities must be such that pL < pX < pR")

    def fSkewNormal(theta):
        loc, scale, a = theta
        return sum(np.array([wL, wX, wR]) * (
                skewnorm.ppf([pL, pX, pR], loc=loc, scale=scale, a=a) - np.array([x0 - uL, x0, x0 + uR])) ** 2)

    try:
        if abs(pR - pL) < 0.75:
            initial_guess = [x0, (uL + uR) / 2, 2]  # Initial guesses of the parameters of the skew-normal distribution
        else:
            initial_guess = [x0, (uL + uR) / 4, 2]

        res = scipy.optimize.minimize(fSkewNormal, initial_guess, method='Nelder-Mead')
        theta = res.x  # Value of parameters of the skew-normal distribution with which it attains a minimum
        return dict(zip(['loc', 'scale', 'a'], theta))
    except:
        raise ValueError("Optimization failed")


def pdf_values(x0, uL, uR, factor=4, num: int = 1000):
    # Make a grid of values around the nominal values, and calculate the pdf for those values

    # If the error bars are equal, we have a normal distribution
    if uL == uR:
        x = np.linspace(x0 - factor * uL, x0 + factor * uR, num)
        y = norm.pdf(x, loc=x0, scale=uL)

    # If the error bars are not equal, we have a skew-normal distribution
    if uL != uR:
        res = parSkewNormal(x0=x0, uL=uL, uR=uR)
        x = np.linspace(res['loc'] - factor * res['scale'], res['loc'] + factor * res['scale'], num)
        y = skewnorm.pdf(x, a=res['a'], loc=res['loc'], scale=res['scale'])

    return x, y


def pdf_value(x, x0, uL, uR):
    # If the error bars are equal, we have a normal distribution
    if uL == uR:
        y = norm.pdf(x, loc=x0, scale=uL)

    # If the error bars are not equal, we have a skew-normal distribution
    if uL != uR:
        res = parSkewNormal(x0=x0, uL=uL, uR=uR)
        y = skewnorm.pdf(x, a=res['a'], loc=res['loc'], scale=res['scale'])

    return y


def draw_samples(x0, uL, uR, size=1000):
    # If the error bars are equal, we have a normal distribution
    if uL == uR:
        samples = norm.rvs(loc=x0, scale=uL, size=size)

    # If the error bars are not equal, we have a skew-normal distribution
    if uL != uR:
        res = parSkewNormal(x0=x0, uL=uL, uR=uR)
        samples = skewnorm.rvs(a=res['a'], loc=res['loc'], scale=res['scale'], size=size)

    return samples


def calculate_lnprior(timing_model, VLBI_data_file, PSR_name: str) -> float:
    # Given the values of RAJ, DECJ, PMRA, PMDEC, PX that we have inserted into the timing model, calculate where they
    # fall in the PDF distributions from the VLBI values
    VLBI_data = pd.read_csv(VLBI_data_file, index_col=0)

    # ------------------------------RAJ------------------------------
    timing_RAJ = Angle(timing_model.RAJ.quantity).rad
    VLBI_RAJ = ufloat(Angle(VLBI_data.loc[PSR_name, "ra_v"]).rad, Angle(VLBI_data.loc[PSR_name, "ra_ve"]).rad)

    RAJ_prior = pdf_value(x=timing_RAJ, x0=VLBI_RAJ.nominal_value, uL=VLBI_RAJ.std_dev, uR=VLBI_RAJ.std_dev)

    # ------------------------------DECJ-----------------------------
    timing_DECJ = Angle(timing_model.DECJ.quantity).rad
    VLBI_DECJ = ufloat(Angle(VLBI_data.loc[PSR_name, "dec_v"]).rad, Angle(VLBI_data.loc[PSR_name, "dec_ve"]).rad)

    DECJ_prior = pdf_value(x=timing_DECJ, x0=VLBI_DECJ.nominal_value, uL=VLBI_DECJ.std_dev, uR=VLBI_DECJ.std_dev)

    # ------------------------------Proper Motion------------------------------
#    VLBI_DECJ = ufloat(Angle(VLBI_data.loc[PSR_name, "VLBI_DECJ"]).rad, Angle(VLBI_data.loc[PSR_name, "VLBI_DECJ_err"]).rad)

    # For VLBI, sometimes the error bars are asymmetric. In order to propagate errors, we will do this twice, each time
    # assuming a symmetric error equal to either VLBI_uL or VLBI_uR:
    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(VLBI_data.loc[PSR_name, "pmra_v"], VLBI_data.loc[PSR_name, "pmra_v_" + error_side])
        VLBI_PMDEC = ufloat(VLBI_data.loc[PSR_name, "pmdec_v"], VLBI_data.loc[PSR_name, "pmdec_v_" + error_side])
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2)
#        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2 * (umath.cos(VLBI_DECJ) ** 2))

        if error_side == "uL":
            VLBI_PM_uL = VLBI_PM.std_dev
        elif error_side == "uR":
            VLBI_PM_uR = VLBI_PM.std_dev

    # Calculate the total proper motion from the timing model
    timing_PMRA = ufloat(timing_model.PMRA.value, timing_model.PMRA.uncertainty.value)
    timing_PMDEC = ufloat(timing_model.PMDEC.value, timing_model.PMDEC.uncertainty.value)
#    timing_DECJ = ufloat(Angle(timing_model.DECJ.quantity).rad, Angle(timing_model.DECJ.uncertainty).rad)
#    timing_PM = umath.sqrt(timing_PMDEC ** 2 + timing_PMRA ** 2 * (umath.cos(timing_DECJ) ** 2))
    timing_PM = umath.sqrt(timing_PMDEC ** 2 + timing_PMRA ** 2)

    # Calculate the prior for the timing value of the PM, given the PDF from the VLBI values
    PM_prior = pdf_value(x=timing_PM.nominal_value, x0=VLBI_PM.nominal_value, uL=VLBI_PM_uL, uR=VLBI_PM_uR)

    # ------------------------------Parallax------------------------------
    PX_prior = pdf_value(x=timing_model.PX.quantity.value, x0=VLBI_data.loc[PSR_name, "px_v"],
                         uL=VLBI_data.loc[PSR_name, "px_v_uL"], uR=VLBI_data.loc[PSR_name, "px_v_uR"])

    # Calculate the joint probability distribution by multiplying the PDFs
#    return np.outer(PX_prior, PM_prior)
    print("RAJ prior = " + str(RAJ_prior))
    print("DECJ prior = " + str(DECJ_prior))
    print("PM_prior = " + str(PM_prior))
    print("PX_prior = " + str(PX_prior))

    if any(x <= 0 for x in [RAJ_prior, DECJ_prior, PM_prior, PX_prior]):
        return 0.0
    return ln(RAJ_prior) + ln(DECJ_prior) + ln(PM_prior) + ln(PX_prior)


def replace_params(timing_model: TimingModel, new_timing_solution: pandas) -> TimingModel:
    # We build a dictionary with a key for each parameter we want to set.
    # The dictionary entries can be either
    #  {'pulsar name': (parameter value, TEMPO_Fit_flag, uncertainty)} akin to a TEMPO par file form
    # or
    # {'pulsar name': (parameter value, )} for parameters that can't be fit
    params = {
        "POSEPOCH": (Time(new_timing_solution.POSEPOCH, format="mjd", scale="tdb"),),
        "RAJ": (new_timing_solution.RAJ, 1, 0 * pint.hourangle_second),
        "DECJ": (new_timing_solution.DECJ, 1, 0 * u.arcsec),
        "PMRA": (new_timing_solution.PMRA, 1, 0 * timing_model.PMRA.units),
        "PMDEC": (new_timing_solution.PMDEC, 1, 0 * timing_model.PMDEC.units),
        "PX": (new_timing_solution.PX, 1, 0 * timing_model.PX.units)
    }

    og_POSEPOCH = Time(timing_model.POSEPOCH.value, format='mjd', scale='tdb')
    og_RAJ = Angle(timing_model.RAJ.quantity, unit=u.hourangle).to_string(unit=u.hourangle, sep=':')
    og_DECJ = Angle(timing_model.DECJ.quantity, unit=u.degree).to_string(unit=u.degree, sep=':')
    og_PMRAJ = timing_model.PMRA.quantity
    og_PMDEC = timing_model.PMDEC.quantity
    og_PX = timing_model.PX.quantity

    # Assign the new parameters
    for name, info in params.items():
        par = getattr(timing_model, name)  # Get parameter object from name
        par.value = info[0]  # set parameter value
        if len(info) > 1:
            if info[1] == 1:
                par.frozen = True  # Frozen means not fit.
            par.uncertainty = info[2]  # set parameter uncertainty

    # Set up and validate the new model
    timing_model.setup()
    timing_model.validate()

    print(f"POSPEOCH before = {og_POSEPOCH}")
    print(f"POSPEOCH after  = {Time(timing_model.POSEPOCH.value, format='mjd', scale='tdb')}")
    print(" ")
    print(f"RAJ before = {og_RAJ}")
    print(f"RAJ after  = {Angle(timing_model.RAJ.quantity, unit=u.hourangle).to_string(unit=u.hourangle, sep=':')}")
    print(" ")
    print(f"DECJ before = {og_DECJ}")
    print(f"DECJ after  = {Angle(timing_model.DECJ.quantity, unit=u.degree).to_string(unit=u.degree, sep=':')}")
    print(" ")
    print(f"PMRA before = {og_PMRAJ}")
    print(f"PMRA after  = {timing_model.PMRA.quantity}")
    print(" ")
    print(f"PMDEC before = {og_PMDEC}")
    print(f"PMDEC after  = {timing_model.PMDEC.quantity}")
    print(" ")
    print(f"PX before = {og_PX}")
    print(f"PX after  = {timing_model.PX.quantity}")

    return timing_model

def add_noise_params(tm: TimingModel, EFAC, EQUAD) -> TimingModel:

    # Add the EFAC, EQUAD components
    from pint.models.noise_model import ScaleToaError
    tm.add_component(ScaleToaError(), validate=False)

    # Add parameter values
    params = {
        "EFAC1": (EFAC, 1, 0),
        "EQUAD1": (EQUAD, 1, 0),
    }

    # Assign the parameters
    for name, info in params.items():
        par = getattr(tm, name)  # Get parameter object from name
        par.quantity = info[0]  # set parameter value
        if len(info) > 1:
            if info[1] == 1:
                par.frozen = False  # Frozen means not fit.
            par.uncertainty = info[2]

    # Set up and validate the model
    tm.setup()
    tm.validate()

    return tm


def unfreeze_noise(mo, verbose=False):
    """
    Unfreeze noise parameters in place in preparation for PINT noise

    Parameters
    ==========
    mo: PINT timing model

    Returns
    =======
    None
    """

    EFAC_EQUAD_components = mo.components['ScaleToaError']
    ECORR_components = mo.components['EcorrNoise']
    # Get the EFAC and EQUAD keys. Ignore TNEQ
    EFAC_keys = EFAC_EQUAD_components.EFACs.keys()
    EQUAD_keys = EFAC_EQUAD_components.EQUADs.keys()
    # Get the ECORR keys
    ECORR_keys = ECORR_components.ECORRs.keys()

    # Iterate over each set and mark unfrozen
    for key in EFAC_keys:
        param = getattr(EFAC_EQUAD_components, key)
        if verbose:
            print("Unfreezing", key)
        param.frozen = False
#        param.quantity += 4.0
    for key in EQUAD_keys:
        param = getattr(EFAC_EQUAD_components, key)
        if verbose:
            print("Unfreezing", key)
        param.frozen = False
    for key in ECORR_keys:
        param = getattr(ECORR_components, key)
        if verbose:
            print("Unfreezing", key)
        param.frozen = False

    # Unfreeze red noise if present
    # if 'PLRedNoise' in mo.components.keys():
    #    mo.components['PLRedNoise'].RNAMP.frozen = False
    #    mo.components['PLRedNoise'].RNIDX.frozen = False

def Wang_frame_tie(VLBI_pos_ICRF_spherical, Omega, astropy):

     # Transform the (RA,DEC) to cartesian components in the ICRF. Do the error propagation automatically.
    # For AstroPy, see https://docs.astropy.org/en/latest/api/astropy.coordinates.spherical_to_cartesian.html
    VLBI_pos_ICRF_xyz = umath_spherical_to_cartesian(VLBI_pos_ICRF_spherical)
    x, y, z = spherical_to_cartesian(1.0, VLBI_pos_ICRF_spherical['dec'].nominal_value, VLBI_pos_ICRF_spherical['ra'].nominal_value)

    # Transform to xyz coordinates from the ICRF to the SSB frame
    matrix = np.matmul(Omega, np.array(list(VLBI_pos_ICRF_xyz.values())))
    VLBI_pos_SSB_xyz = dict(zip(['x', 'y', 'z'], matrix))
    SSB_x, SSB_y, SSB_z = np.array(np.dot(Omega, np.array([x, y, z])))

    # Transform cartesian components in the SSB frame to (RA,DEC)
    VLBI_pos_SSB_spherical = umath_cartesian_to_spherical(VLBI_pos_SSB_xyz)
    r, dec, ra = cartesian_to_spherical(SSB_x, SSB_y, SSB_z)  # ra and dec are returned in radians

    if astropy:
        return dict(ra=ufloat(ra.value, VLBI_pos_SSB_spherical["ra"].std_dev),
                    dec=ufloat(dec.value, VLBI_pos_SSB_spherical["dec"].std_dev))
    else:
        return dict(ra=ufloat(VLBI_pos_SSB_spherical["ra"].nominal_value, VLBI_pos_SSB_spherical["ra"].std_dev),
                    dec=ufloat(VLBI_pos_SSB_spherical["dec"].nominal_value, VLBI_pos_SSB_spherical["dec"].std_dev))


def epoch_scrunch(toas, data=None, errors=None, epochs=None, decimals=0, getdict=False, weighted=False, harmonic=False):
    if epochs is None:
        epochsize = 10 ** (-decimals)
        bins = np.arange(np.around(min(toas), decimals=decimals) - epochsize,
                         np.around(max(toas), decimals=decimals) + 2 * epochsize,
                         epochsize)  # 2 allows for the extra bin to get chopped by np.histogram
        freq, bins = np.histogram(toas, bins)
        validinds = np.where(freq != 0)[0]

        epochs = np.sort(bins[validinds])
        diffs = np.array(list(map(lambda x: np.around(x, decimals=decimals), np.diff(epochs))))
        epochs = np.append(epochs[np.where(diffs > epochsize)[0]], [epochs[-1]])
    else:
        epochs = np.array(epochs)
    reducedTOAs = np.array(list(map(lambda toa: epochs[np.argmin(np.abs(epochs - toa))], toas)))

    if data is None:
        return epochs

    Nepochs = len(epochs)

    if weighted and errors is not None:
        averaging_func = lambda x, y: weighted_moments(x, 1.0 / y ** 2, unbiased=True, harmonic=harmonic)
    else:
        averaging_func = lambda x, y: (np.mean(x), np.std(y))  # is this correct?

    if getdict:
        retval = dict()
        retvalerrs = dict()
    else:
        retval = np.zeros(Nepochs)
        retvalerrs = np.zeros(Nepochs)
    for i in range(Nepochs):
        epoch = epochs[i]
        inds = np.where(reducedTOAs == epoch)[0]
        if getdict:
            retval[epoch] = data[inds]
            if errors is not None:
                retvalerrs[epoch] = errors[inds]
        else:
            if errors is None:
                retval[i] = np.mean(data[inds])  # this is incomplete
                retvalerrs[i] = np.std(data[inds])  # temporary
            else:
                retval[i], retvalerrs[i] = averaging_func(data[inds], errors[inds])
    #            print data[inds],errors[inds]
    if getdict and errors is None:  # is this correct?
        return epochs, retval
    return epochs, retval, retvalerrs
