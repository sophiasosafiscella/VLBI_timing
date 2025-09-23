import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, ICRS, spherical_to_cartesian, cartesian_to_spherical, Angle
import astropy.units as u
from math import sqrt, sin, cos
import uncertainties
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import statsmodels.api as sm
import sys


def to_uarray(array):
    nominal_values = [array[i].nominal_value for i in range(len(array))]
    std_devs = [array[i].std_dev for i in range(len(array))]
    return uarray(nominal_values, std_devs)


def to_ufloat(x):
    return ufloat(x.nominal_value, x.std_dev)


def diff_pos(ra: float, dec: float):
    return  [[-sin(ra) * cos(dec), -1.0 * cos(ra) * sin(dec)], [cos(ra) * cos(dec), -1.0 * sin(ra) * sin(dec)], [0, cos(dec)]]


#---------------------------------------------
# Read the position of VLBI calibration source
#---------------------------------------------
path: str = './data/NG_frame_tie/'
cal = pd.read_table(path + 'NG_cal.csv', header=0, index_col=0, sep=',', comment='#')

# Original positions
og_cat = cal["og_cat"]
og_cal_ra = uarray(Angle(cal["og_cal_ra"], unit=u.hourangle).rad, Angle(cal["og_cal_rae"], unit=u.hourangle).rad)
og_cal_dec = uarray(Angle(cal["og_cal_dec"], unit=u.degree).rad, Angle(cal["og_cal_dece"], unit=u.arcsec).rad)

# Positions in RFC
rfc_cal_ra = uarray(Angle(cal["RFC_cal_ra"], unit=u.hourangle).rad, Angle(cal["RFC_cal_rae"], unit=u.hourangle).rad)
rfc_cal_dec = uarray(Angle(cal["RFC_cal_dec"], unit=u.degree).rad, Angle(cal["RFC_cal_dece"], unit=u.arcsec).rad)

# Calculate the offsets between the original positions and RFC
dcal_ra = rfc_cal_ra - og_cal_ra
dcal_dec = rfc_cal_dec - og_cal_dec

#---------------------------------------------
# Read in the pulsar VLBI positions in their original catalogues
#---------------------------------------------
vlbi_pos = pd.read_table(path + 'NG_msp_vlbi.csv', header=0, index_col=0, sep=',', comment='#')
psr_names = vlbi_pos.index.tolist()
N_pulsars: int = len(psr_names)

ra_v = uarray(Angle(vlbi_pos["ra_v"], unit=u.hourangle).rad, Angle(vlbi_pos["ra_ve"], unit=u.hourangle).rad)  # RA from VLBI
dec_v = uarray(Angle(vlbi_pos["dec_v"], unit=u.degree).rad, Angle(vlbi_pos["dec_ve"], unit=u.arcsec).rad)     # DEC from VLBI

ra_v2 = np.empty(len(ra_v), dtype=uncertainties.core.Variable)
dec_v2 = np.empty(len(dec_v), dtype=uncertainties.core.Variable)

# Calibrate positions to RFC
for i, (index, row) in enumerate(cal.iterrows()):
    if row['og_cat'] != "RFC":
        print("Calibrating " + index + " to RFC")
        ra_v2[i] = to_ufloat(ra_v[i] + dcal_ra[i])     # RA from VLBI in RFC
        dec_v2[i] = to_ufloat(dec_v[i] + dcal_dec[i])  # DEC from VLBI in RFC
    else:
        ra_v2[i] = ra_v[i]
        dec_v2[i] = dec_v[i]

# change the error bar of 0437 and do not correct the uncertainties for the other pulsars
for i in range(N_pulsars):
   if vlbi_pos.index.tolist()[i] == 'J0437-4715':
       ra_v2[i].std_dev = sqrt(ra_v[i].std_dev**2 + rfc_cal_ra[i].std_dev**2 + Angle(0.8 * u.mas).rad**2)
       dec_v2[i].std_dev = sqrt(dec_v[i].std_dev**2 + rfc_cal_dec[i].std_dev**2)
   else:
       ra_v2[i].std_dev = ra_v[i].std_dev       # "We did not correct the published uncertainties to those in ICRF2"
       dec_v2[i].std_dev = dec_v[i].std_dev

# Turn the VLBI positions into dictionaries
ra0 = {k: v for k, v in zip(psr_names, ra_v2)}    # This seems to agree with Wang's
dec0 = {k: v for k, v in zip(psr_names, dec_v2)}  # This seems to agree with Wang's

#---------------------------------------------
# Read in timing positions
#---------------------------------------------
timing_pos = pd.read_table(path + 'NG_msp_timing.csv', header=0, index_col=0, sep=',', comment='#')
psr_t_names = timing_pos.index.tolist()
psr_list = timing_pos.index.unique().tolist()

# Make sure that the epochs are consistent between VLBI and timing
for PSR in psr_list:
    if timing_pos.loc[PSR, 'epoch_t'] != vlbi_pos.loc[PSR, "epoch_v"]:
        print('WARNING vlbi and timing epochs are not consistent')
        sys.exit(1)

for j, ephem in enumerate(timing_pos['ephem'].unique()):
    B = np.zeros((3 * N_pulsars, 3))
    D = np.zeros((3 * N_pulsars, 2 * N_pulsars))
    Ct, Cv = np.zeros((2 * N_pulsars, 2 * N_pulsars)), np.zeros((2 * N_pulsars, 2 * N_pulsars))

    rat_hms= timing_pos.loc[timing_pos['ephem'] == ephem, 'ra_t']
    rat_dict = {k: v for k, v in zip(rat_hms.index.tolist(), Angle(rat_hms.values, unit=u.hourangle).rad)}
    rat_diff = Angle([rat_dict[psr] - ra0[psr].nominal_value for psr in psr_list], unit=u.rad).to(u.mas).value

    dect_dms = timing_pos.loc[timing_pos['ephem'] == ephem, 'dec_t']
    dect_dict = {k: v for k, v in zip(dect_dms.index.tolist(), Angle(dect_dms.values, unit=u.degree).rad)}
    dect_diff = Angle([dect_dict[psr] - dec0[psr].nominal_value for psr in psr_list], unit=u.rad).to(u.mas).value

    print(rat_diff)
    print(dect_diff)

    rat_err = timing_pos.loc[timing_pos['ephem'] == ephem, 'ra_te']
    rat_err = {k: v for k, v in zip(rat_err.index.tolist(), Angle(rat_err.values, unit=u.hourangle).to(u.mas).value)}

    dect_err = timing_pos.loc[timing_pos['ephem'] == ephem, 'dec_te']
    dect_err = {k: v for k, v in zip(dect_err.index.tolist(), Angle(dect_err.values, unit=u.deg).to(u.mas).value)}

    cor = timing_pos.loc[timing_pos['ephem'] == ephem, 'cor']
    rdcorc = {k: v for k, v in zip(cor.index.tolist(), cor.values)}

    for i, PSR in enumerate(psr_names):
        nhat = spherical_to_cartesian(1.0, dec0[PSR].nominal_value, ra0[PSR].nominal_value)
        B[3 * i: 3 * (i + 1),:] = np.matrix([[0,-nhat[2],nhat[1]], [nhat[2],0,-nhat[0]], [-nhat[1],nhat[0],0]])
        D[3 * i:3 * (i + 1), 2 * i:2 * (i + 1)] = diff_pos(ra0[PSR].nominal_value, dec0[PSR].nominal_value)

        # This seems to be some sort of covariance matrix. I think it's equivalent to Sigma from Dusty's paper,
        # because it includes rdcorc, which is the "dimensionless normalized cross-correlation between RA and Dec."
        Ct[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = np.matrix([[rat_err[PSR] ** 2, rat_err[PSR] * dect_err[PSR] * rdcorc[PSR]],
                                                              [rat_err[PSR] * dect_err[PSR] * rdcorc[PSR], dect_err[PSR] ** 2]])

        # VLBI covariance matrix?
        Cv[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = np.diag([Angle(ra0[PSR].std_dev, unit=u.rad).to(u.mas).value ** 2,
                                                            Angle(dec0[PSR].std_dev, unit=u.rad).to(u.mas).value ** 2])

    # now set up the LSQ dEq = M A + eps, where M = (Dt*D)^-1*Dt
    cov_matrix = Ct + Cv                                                                # This agrees with Wang's
    M = np.linalg.multi_dot([np.linalg.inv(np.matmul(D.T, D)), D.T, B])                 # This agrees with Wang's
    data = np.array([(rat_diff[i], dect_diff[i]) for i in range(N_pulsars)]).flatten()  # Data is not the same as Wang's

    gls_model = sm.GLS(data, M, sigma=cov_matrix).fit()
    print(gls_model.summary())
#    print([gls_model.params])
    print(gls_model.cov_params())

    pd.DataFrame([gls_model.params], columns=['Ax', 'Ay', 'Az']).to_csv("./data/NG_frame_tie/NG_frame_tie.csv", index=False)
