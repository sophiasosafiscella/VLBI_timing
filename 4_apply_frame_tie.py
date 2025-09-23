import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord, ICRS
from astropy.time import Time
import astropy.units as u

from VLBI_utils import Wang_frame_tie
from uncertainties import ufloat

VLBI_data = pd.read_csv("./data/calibrated_vlbi_astrometric_data.csv", header=0, index_col=0)
PSR_list = VLBI_data.index

correct_proper_motion: bool = False

# Frame tie
A = pd.read_csv('./data/NG_frame_tie/NG_frame_tie.csv', header=0).to_dict('records')[0]

# Make sure Omega is in radians and not mas
Ax, Ay, Az = Angle(A['Ax'], u.mas).rad, Angle(A['Ay'], u.mas).rad, Angle(A['Az'], u.mas).rad
Omega = np.array([[1.0, Az, -1.0 * Ay], [-1.0 * Az, 1.0, Ax], [Ay, -1.0 * Ax, 1.0]])

for i, PSR in enumerate(PSR_list):

    print("Processing PSR", PSR)

    # ----------------------------------------------------------------------------------------------
    # VLBI frame tie for position
    # ----------------------------------------------------------------------------------------------

    VLBI_pos_ICRF = SkyCoord(ra=VLBI_data.loc[PSR, "ra_v"], dec=VLBI_data.loc[PSR, "dec_v"],
                             frame=ICRS, unit=(u.hourangle, u.deg),
                             equinox=VLBI_data.loc[PSR, "equinox"],
                             obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    VLBI_pos_ICRF_err = SkyCoord(ra=VLBI_data.loc[PSR, "ra_ve"], dec=VLBI_data.loc[PSR, "dec_ve"],
                             frame=ICRS, unit=(u.hourangle, u.deg),
                             equinox=VLBI_data.loc[PSR, "equinox"],
                             obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    # Create uncertainty objects to handle error propagation
    VLBI_pos_ICRF_spherical = dict(ra=ufloat(VLBI_pos_ICRF.ra.rad, VLBI_pos_ICRF_err.ra.rad),
                                dec=ufloat(VLBI_pos_ICRF.dec.rad, VLBI_pos_ICRF_err.dec.rad))

    # Apply the frame tie
    VLBI_pos_SBB_spherical = Wang_frame_tie(VLBI_pos_ICRF_spherical, Omega, astropy=True)

    # Create SkyCoord objects for the positions in the SSB system
    VLBI_pos_SSB = SkyCoord(ra=VLBI_pos_SBB_spherical["ra"].nominal_value, dec=VLBI_pos_SBB_spherical["dec"].nominal_value,
                            frame=ICRS, unit=(u.rad, u.rad),
                            equinox=VLBI_data.loc[PSR, "equinox"],
                            obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    VLBI_pos_SSB_err = SkyCoord(ra=VLBI_pos_SBB_spherical["ra"].std_dev, dec=VLBI_pos_SBB_spherical["dec"].std_dev,
                            frame=ICRS, unit=(u.rad, u.rad),
                            equinox=VLBI_data.loc[PSR, "equinox"],
                            obstime=Time(val=VLBI_data.loc[PSR, "epoch_v"], format='mjd', scale='utc'))

    # Output the corrected positions
    VLBI_data.loc[PSR, "ra_v"] = Angle(VLBI_pos_SSB.ra).to_string(unit=u.hourangle)
    VLBI_data.loc[PSR, "dec_v"] = VLBI_pos_SSB.dec.to_string(unit=u.degree)
    VLBI_data.loc[PSR, "ra_ve"] = VLBI_pos_SSB_err.ra.to_string(unit=u.hourangle)
    VLBI_data.loc[PSR, "dec_ve"] = VLBI_pos_SSB_err.dec.to_string(unit=u.degree)

    # ----------------------------------------------------------------------------------------------
    # VLBI frame tie for proper motion
    # ----------------------------------------------------------------------------------------------
    if correct_proper_motion:
        for error_side in ["uL", "uR"]:
            VLBI_PM_ICRF = dict(PMRA=Angle(VLBI_data.loc[PSR, "pmra_v"], unit=u.mas),
                                PMDEC=Angle(VLBI_data.loc[PSR, "pmdec_v"], unit=u.mas))

            VLBI_PM_ICRF_err = dict(PMRA=Angle(VLBI_data.loc[PSR, f"pmra_v_{error_side}"], unit=u.mas),
                                    PMDEC=Angle(VLBI_data.loc[PSR, f"pmdec_v_{error_side}"], unit=u.mas))

            # Create uncertainty objects to handle error propagation
            VLBI_PM_ICRF_spherical = dict(ra=ufloat(VLBI_PM_ICRF['PMRA'].rad, VLBI_PM_ICRF_err['PMRA'].rad),
                                          dec=ufloat(VLBI_PM_ICRF['PMDEC'].rad, VLBI_PM_ICRF_err['PMDEC'].rad))

            # Apply the frame tie
            VLBI_PM_SBB_spherical = Wang_frame_tie(VLBI_PM_ICRF_spherical, Omega = np.identity(3), astropy=False)

            if error_side == "uL":
                VLBI_PMRA_SSB_uL = VLBI_PM_SBB_spherical["ra"].std_dev
                VLBI_PMDEC_SSB_uL = VLBI_PM_SBB_spherical["dec"].std_dev
            elif error_side == "uR":
                VLBI_PMRA_SSB_uR = VLBI_PM_SBB_spherical["ra"].std_dev
                VLBI_PMDEC_SSB_uR = VLBI_PM_SBB_spherical["dec"].std_dev

        VLBI_PM_SSB = dict(PMRA=Angle(VLBI_PM_SBB_spherical["ra"].nominal_value, unit=u.rad).to(u.mas).value,
                            PMDEC=Angle(VLBI_PM_SBB_spherical["dec"].nominal_value, unit=u.rad).to(u.mas).value)

        VLBI_PM_SSB_err = dict(PMRA_uL=Angle(VLBI_PMRA_SSB_uL, unit=u.rad).to(u.mas).value,
                               PMRA_uR=Angle(VLBI_PMRA_SSB_uR, unit=u.rad).to(u.mas).value,
                               PMDEC_uL=Angle(VLBI_PMDEC_SSB_uL, unit=u.rad).to(u.mas).value,
                               PMDEC_uR=Angle(VLBI_PMDEC_SSB_uR, unit=u.rad).to(u.mas).value)

        # Output the corrected proper motions
        VLBI_data.loc[PSR, "pmara_v"] = Angle(VLBI_pos_SSB.ra).to_string(unit=u.hourangle)
        VLBI_data.loc[PSR, "pmdec_v"] = VLBI_pos_SSB.dec.to_string(unit=u.degree)
        VLBI_data.loc[PSR, "pmra_ve"] = VLBI_pos_SSB_err.ra.to_string(unit=u.hourangle)
        VLBI_data.loc[PSR, "pmdec_ve"] = VLBI_pos_SSB_err.dec.to_string(unit=u.degree)

VLBI_data.to_csv("./data/frame_tied_vlbi_astrometric_data.csv")
