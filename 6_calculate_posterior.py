#----------------------------------------------------------------------------------------------------------------------
# Calculate the posterior for each trial astrometric solution
#----------------------------------------------------------------------------------------------------------------------

import glob
import os
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint.fitter
import seaborn as sns
from astropy.time import Time
from pint.models import get_model
from pint.residuals import Residuals
from pint.toa import get_TOAs
from pint_pal import noise_utils
from uncertainties import unumpy

from VLBI_utils import calculate_lnprior, replace_params, epoch_scrunch


def calculate_post(PSR_name: str, timing_solution, timfile: str, parfile: str, VLBI_astrometric_data_file, resume=True, plot=False):
    sns.set_theme(context="paper", style="darkgrid", font_scale=1.5)

    print(f"Processing iteration {timing_solution.Index} of {PSR_name}")

    chains_dir : str = f"./noisemodel_linear_sd/{PSR_name}/timing_solution_{timing_solution.Index}/"
    new_par_dir: str = "./results/new_fits/" + PSR_name
    if not os.path.exists(new_par_dir):
        os.makedirs(new_par_dir)
    new_par_file: str = new_par_dir + "/solution_" + str(timing_solution.Index) + "_new.par"

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    original_epoch = Time(ec_timing_model.POSEPOCH.value, format='mjd', scale='tdb')
    eq_timing_model = ec_timing_model.as_ICRS(epoch=original_epoch)

    # Load the TOAs
    toas = get_TOAs(timfile, planets=True, ephem=eq_timing_model.EPHEM.value)

    # Plot the original timing residuals
    if plot:
        # Calculate residuals. Don't forget to use actual timing residuals!
        residuals = Residuals(toas, eq_timing_model).time_resids.to(u.us).value
        xt = toas.get_mjds()
        errors = toas.get_errors().to(u.us).value

        plt.figure()
        plt.errorbar(xt, residuals, yerr=errors, fmt='o')
        plt.title(str(PSR_name) + " Original Timing Residuals | $\sigma_\mathrm{TOA}$ = " + str(
            round(np.std(residuals), 2)))
        plt.xlabel("MJD")
        plt.ylabel("Residual ($\mu s$)")
        plt.tight_layout()
        plt.savefig("./results/new_fits/" + PSR_name + "/" + str(timing_solution.Index) + "_pre.png")
        plt.show()

    # Unfreeze the EFAC and EQUAD noise parameters
    #        unfreeze_noise(eq_timing_model)   # Thanks, Michael!

    # Replace the timing parameter values in the model with those from the new timing solution
    eq_timing_model = replace_params(eq_timing_model, timing_solution)

#    # Change the reference epoch of the timing model to go back to the original epoch, since the TOAs are reference to
#    # the original epoch and not to the new epoch that was used to calculate the overlap between VLBI and timing
#    eq_timing_model.change_posepoch(original_epoch)

    if os.path.exists(new_par_file) and resume:
        final_fit_model = get_model(new_par_file)  # Ecliptical coordiantes
        final_fit_resids = pint.residuals.Residuals(toas=toas, model=final_fit_model)
        print("New model loaded from file")

    else:
        # Perform initial fit
        print("Performing the initial fit...")
        initial_fit = pint.fitter.Fitter.auto(toas, eq_timing_model)
        try:
            initial_fit.fit_toas()
            print("Initial fit done.")
            refitted_timing_model = initial_fit.model
        except:
            print("Fitting new timing solution failed")
            return initial_fit.resids.lnlikelihood()

        # Re-run noise
        print("Re-running noise")
        if not os.path.exists(chains_dir):
            os.mkdir(chains_dir)

        try:
            noise_utils.model_noise(refitted_timing_model, toas, vary_red_noise=True, n_iter=int(5e4), using_wideband=False,
                                    resume=resume, run_noise_analysis=True, base_op_dir=chains_dir)
            newmodel = noise_utils.add_noise_to_model(refitted_timing_model, save_corner=False, base_dir=chains_dir)
            print("Done!")
        except:
            print("Re-running noise failed")
            return initial_fit.resids.lnlikelihood()

        # Final fit
        final_fit = pint.fitter.DownhillGLSFitter(toas, newmodel)

        try:
            print("Fitting the new model")
            final_fit.fit_toas()
            final_fit_model = final_fit.model
            final_fit_model.write_parfile(new_par_file)  # Save the new .par fil
            final_fit_resids = final_fit.resids
            print("New model fitting done.")

        except:
            print("Fitting new timing solution failed")
            return final_fit.resids.lnlikelihood()

    # Get the new residuals
    print("Calculating the new residuals")
    new_res_avg_dict = final_fit_resids.ecorr_average(use_noise_model=True)
    new_res_avg = new_res_avg_dict['time_resids'].to(u.us).value
    new_res_avg_errs = new_res_avg_dict['errors'].to(u.us).value
    new_res_avg_mjds = new_res_avg_dict['mjds'].value

    # Average the observations at different frequencies within each time window
    new_res_epochs, new_res_avg_residuals, maxpost_avg_errors = epoch_scrunch(new_res_avg_mjds,
                                                                              data=new_res_avg,
                                                                              errors=new_res_avg_errs,
                                                                              weighted=True)

    new_res = unumpy.uarray(new_res_avg_residuals, maxpost_avg_errors)

    # Take the difference in the residuals
    ng15_res = np.load(posteriors_dir + "/ng15_res.npy", allow_pickle=True)
    res_diff = ng15_res - new_res

    # Calculate the posterior for this model and TOAs
    ln_prior = calculate_lnprior(final_fit_model, VLBI_astrometric_data_file, PSR_name)
    ln_likelihood = final_fit_resids.lnlikelihood()
    ln_posterior = ln_prior + ln_likelihood
    posterior = ln_posterior
    #        posterior = exp(ln_posterior)
    print("Log(Prior) = " + str(ln_prior))
    print("Log(Likelihood) = " + str(ln_likelihood))
    print("Log(Posterior) = " + str(ln_posterior))

    # Let's plot the residuals and compare
    if plot:
        plt.figure()
        plt.errorbar(
            xt,
            final_fit_resids.time_resids.to(u.us).value,
            toas.get_errors().to(u.us).value,
            fmt='o',
        )
        plt.title("%s Post-Fit Timing Residuals" % PSR_name)
        plt.xlabel("MJD")
        plt.ylabel("Residual ($\mu s$)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("./results/new_fits/" + PSR_name + "/" + str(timing_solution.Index) + "_post.png")
        plt.show()

    return posterior, res_diff, new_res_epochs


if __name__ == "__main__":
    PSR_name, idx, RAJ, DECJ, PX, PMRA, PMDEC, POSEPOCH = sys.argv[1:]  # Timing solution index and parameters

    print(PSR_name, idx, RAJ, DECJ, PX, PMRA, PMDEC, POSEPOCH)

    timing_solution_dict = {"Index": idx, "RAJ": RAJ, "DECJ": DECJ, "PX": PX, "PMRA": PMRA, "PMDEC": PMDEC, "POSEPOCH": POSEPOCH}

    # Convert dictionary to DataFrame
    for t in pd.DataFrame(timing_solution_dict, columns=list(timing_solution_dict.keys())[1:], index=[timing_solution_dict['Index']]).itertuples(index=True):
        timing_solution = t

    posteriors_dir: str = f"./results/timing_posteriors_frame_tie/{PSR_name}"
    if not os.path.exists(posteriors_dir):
        os.makedirs(posteriors_dir)

    VLBI_astrometric_data_file: str = "./data/calibrated_vlbi_astrometric_data.csv"

    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}_PINT*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}_PINT*par")[0]

    # Calculate the posterior
    posterior, residuals_diff, res_epochs = calculate_post(PSR_name, timing_solution, timfile, parfile, VLBI_astrometric_data_file, resume=True, plot=False)
    print("Posterior and new residuals calculated. Now saving them...")

    # Save the timing solution with its posterior
    results_df = pd.DataFrame({
        'idx': [idx],
        'POSEPOCH': [POSEPOCH],
        'RAJ': [RAJ],
        'DECJ': [DECJ],
        'PX': [PX],
        'PMRA': [PMRA],
        'PMDEC': [PMDEC],
        'posterior': [posterior],
        'residuals_diff': [residuals_diff],
        'res_epochs': [res_epochs]
    })
    results_df.to_pickle(posteriors_dir + "/" + str(idx) + "_results.pkl")
    print("Posterior saved. End of calculate_posterior.py")
