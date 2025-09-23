import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, ICRS, spherical_to_cartesian, cartesian_to_spherical, Angle
import astropy.units as u
import uncertainties
from uncertainties.unumpy import uarray
from math import sqrt
import sys

#---------------------------------------------
# Read the position of VLBI calibration source
#---------------------------------------------
cal = pd.read_table('./data/NG_frame_tie/NG_cal.csv', header=0, index_col=0, sep=',', comment='#')
cal_psr_list = cal.index.tolist()

# Original positions
og_cat = cal["og_cat"]
og_cal_ra = uarray(Angle(cal["og_cal_ra"], unit=u.hourangle).rad, Angle(cal["og_cal_rae"], unit=u.hourangle).rad)
og_cal_dec = uarray(Angle(cal["og_cal_dec"], unit=u.degree).rad, Angle(cal["og_cal_dece"], unit=u.degree).rad)

# Positions in RFC
rfc_cal_ra = uarray(Angle(cal["RFC_cal_ra"], unit=u.hourangle).rad, Angle(cal["RFC_cal_rae"], unit=u.hourangle).rad)
rfc_cal_ra_dict = {k:v for k, v in zip(cal_psr_list, rfc_cal_ra)}

rfc_cal_dec = uarray(Angle(cal["RFC_cal_dec"], unit=u.degree).rad, Angle(cal["RFC_cal_dece"], unit=u.degree).rad)
rfc_cal_dec_dict = {k:v for k, v in zip(cal_psr_list, rfc_cal_dec)}

# Calculate the offsets between the original positions and RFC
dcal_ra = rfc_cal_ra - og_cal_ra
dcal_ra_dict = {k:v for k, v in zip(cal_psr_list, dcal_ra)}
dcal_dec = rfc_cal_dec - og_cal_dec
dcal_dec_dict = {k:v for k, v in zip(cal_psr_list, dcal_dec)}

#---------------------------------------------------------------
# Read in the pulsar VLBI positions in their original catalogues
#---------------------------------------------------------------
vlbi_pos = pd.read_table('./data/NG_frame_tie/NG_msp_vlbi.csv', header=0, index_col=0, sep=',', comment='#')
#vlbi_pos = pd.read_table('./data/original_vlbi_astrometric_data.csv', header=0, index_col=0, sep=',', comment='#')
NG_psr_list = vlbi_pos.index.tolist()

ra_v = uarray(Angle(vlbi_pos["ra_v"], unit=u.hourangle).rad, Angle(vlbi_pos["ra_ve"], unit=u.hourangle).rad)  # RA from VLBI
dec_v = uarray(Angle(vlbi_pos["dec_v"], unit=u.degree).rad, Angle(vlbi_pos["dec_ve"], unit=u.degree).rad)     # DEC from VLBI

ra_v2 = np.empty(len(ra_v), dtype=uncertainties.core.Variable)
dec_v2 = np.empty(len(dec_v), dtype=uncertainties.core.Variable)

# Calibrate positions to RFC
for psr_name, row in vlbi_pos.iterrows():

    if og_cat[psr_name] != "RFC":
        print(psr_name)
        print(row['ra_v'], row['dec_v'])

        vlbi_pos.loc[psr_name,'ra_v'] = Angle(Angle(row['ra_v'], unit=u.hourangle).rad + dcal_ra_dict[psr_name].nominal_value, unit=u.rad).to_string(unit=u.hourangle)
        vlbi_pos.loc[psr_name,'dec_v'] = Angle(Angle(row['dec_v'], unit=u.deg).rad + dcal_dec_dict[psr_name].nominal_value, unit=u.rad).to_string(unit=u.deg)

        print(vlbi_pos.loc[psr_name,'ra_v'], vlbi_pos.loc[psr_name,'dec_v'])
        print(" ")

    # new info from adam deller
    if psr_name == 'J0437-4715':
        vlbi_pos.loc[psr_name,'ra_ve'] = Angle(sqrt(Angle(row['ra_ve'], unit=u.hourangle).rad**2 + rfc_cal_ra_dict[psr_name].std_dev**2 + Angle(0.8 * u.mas).rad**2), unit=u.rad).to_string(unit=u.hourangle)
        vlbi_pos.loc[psr_name,'dec_ve'] = Angle(sqrt(Angle(row['ra_ve'], unit=u.deg).rad**2 + rfc_cal_dec_dict[psr_name].std_dev**2), unit=u.rad).to_string(unit=u.deg)

vlbi_pos.to_csv('./data/new_calibrated_vlbi_astrometric_data.csv')

