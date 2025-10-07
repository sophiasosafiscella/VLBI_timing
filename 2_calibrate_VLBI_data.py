#----------------------------------------------------------------------------------------------------------------------
# 2) Calibrate the VLBI astrometric positions to the RFC
#----------------------------------------------------------------------------------------------------------------------

from math import sqrt
import astropy.units as u
import pandas as pd
from astropy.coordinates import Angle
from VLBI_utils import handle_error
from uncertainties.unumpy import uarray

#---------------------------------------------
# Read the position of VLBI calibration source
#---------------------------------------------
cal_data = pd.read_table('./data/full_cal_data.csv', header=0, index_col=0, sep=',', comment='#')
cal_psr_list = cal_data.index.tolist()

# Original positions of the calibration sources. Input errors must be in mas.
og_cal_ra = uarray(Angle(cal_data["og_cal_ra"], unit=u.hourangle).rad, Angle(cal_data["og_cal_rae"], unit=u.mas).to(u.rad).value)
og_cal_dec = uarray(Angle(cal_data["og_cal_dec"], unit=u.degree).rad, Angle(cal_data["og_cal_dece"], unit=u.mas).to(u.rad).value)

# Positions in RFC of the calibration sources. Input errors must be in mas.
rfc_cal_ra = uarray(Angle(cal_data["RFC_cal_ra"], unit=u.hourangle).rad, Angle(cal_data["RFC_cal_rae"], unit=u.mas).to(u.rad).value)
rfc_cal_ra_dict = {k:v for k, v in zip(cal_psr_list, rfc_cal_ra)}

rfc_cal_dec = uarray(Angle(cal_data["RFC_cal_dec"], unit=u.degree).rad, Angle(cal_data["RFC_cal_dece"], unit=u.mas).to(u.rad).value)
rfc_cal_dec_dict = {k:v for k, v in zip(cal_psr_list, rfc_cal_dec)}

# Calculate the offsets between the original positions and RFC
dcal_ra = rfc_cal_ra - og_cal_ra
dcal_ra_dict = {k:v for k, v in zip(cal_psr_list, dcal_ra)}
dcal_dec = rfc_cal_dec - og_cal_dec
dcal_dec_dict = {k:v for k, v in zip(cal_psr_list, dcal_dec)}

#---------------------------------------------------------------
# Read in the pulsar VLBI positions in their original catalogues
#---------------------------------------------------------------
vlbi_pos = pd.read_table('./data/msp_vlbi.csv', header=0, index_col=0, sep=',', comment='#')

# Calibrate positions to RFC
for psr_name, row in vlbi_pos.iterrows():

    if cal_data.loc[psr_name]["og_cat"] != "RFC":

        vlbi_pos.loc[psr_name,'ra_v'] = Angle(Angle(row['ra_v'], unit=u.hourangle).rad + dcal_ra_dict[psr_name].nominal_value, unit=u.rad).to_string(unit=u.hourangle)
        vlbi_pos.loc[psr_name,'dec_v'] = Angle(Angle(row['dec_v'], unit=u.deg).rad + dcal_dec_dict[psr_name].nominal_value, unit=u.rad).to_string(unit=u.deg)

    # new info from adam deller
    if psr_name == 'J0437-4715':
        J04_ra_ve = Angle(handle_error(row['ra_ve'], (Angle(row['dec_v'], unit=u.deg).degree)), unit=u.mas)
        J04_dec_ve = Angle(handle_error(row['dec_ve']), unit=u.mas)

        vlbi_pos.loc[psr_name,'ra_ve'] = Angle(sqrt(J04_ra_ve.to(u.rad).value**2 + rfc_cal_ra_dict[psr_name].std_dev**2 + Angle(0.8 * u.mas).to(u.rad).value**2), unit=u.rad).to(u.mas)
        vlbi_pos.loc[psr_name,'dec_ve'] = Angle(sqrt(J04_dec_ve.to(u.rad).value**2 + rfc_cal_dec_dict[psr_name].std_dev**2), unit=u.rad).to(u.mas)

    else:
        vlbi_pos.loc[psr_name, 'ra_ve'] = handle_error(row['ra_ve'], (Angle(row['dec_v'], unit=u.deg).degree))
        vlbi_pos.loc[psr_name, 'dec_ve'] = handle_error(row['dec_ve'])

vlbi_pos.to_csv('./data/vlbi_astrometric_data_calibrated.csv', index=True)
