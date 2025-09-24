#----------------------------------------------------------------------------------------------------------------------
# Make posterior corner plots
#----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns
from astropy.coordinates import Angle
import astropy.units as u
from pint.models import get_model
from scipy.integrate import trapezoid as trapz
import itertools
import glob

def find_best_sol(df):

    # Index of the maximum posterior solution
    best_sol_idx = result_df['posterior'].idxmax()

    # Maximum posterior solution
    best_sol = result_df.loc[best_sol_idx].to_dict()

    # Maximum posterior errorbars
    best_sol_err = dict.fromkeys(df.columns[1:-1])

    # Find the error bars as half the size of a pixel
    for col in df.columns[1:-1]:
        grid = df[col].unique()
        if col == 'RAJ':
            best_sol_err[col] = (Angle(grid[1], unit=u.hourangle) - Angle(grid[0], unit=u.hourangle)).mas/2.0
        elif col == 'DECJ':
            best_sol_err[col] = (Angle(grid[1], unit=u.degree) - Angle(grid[0], unit=u.degree)).mas/2.0
        else:
            best_sol_err[col] = (grid[1]-grid[0])/2.0

    return best_sol, best_sol_err


def delog(df):

    # Step 0: convert strings to floats
    float_df = df.copy()
    float_df['RAJ'] = Angle(float_df['RAJ'].to_numpy(), unit=u.hourangle).to(u.rad).value
    float_df['DECJ'] = Angle(float_df['DECJ'].to_numpy(), unit=u.hourangle).to(u.rad).value

    # Step 1: extract unique grid values
    RAJ_vals = np.sort(float_df['RAJ'].unique())
    DECJ_vals = np.sort(float_df['DECJ'].unique())
    PMRA_vals = np.sort(float_df['PMRA'].unique())
    PMDEC_vals = np.sort(float_df['PMDEC'].unique())
    PX_vals = np.sort(float_df['PX'].unique())

    # Create full MultiIndex from all combinations
    full_index = pd.MultiIndex.from_product(
        [RAJ_vals, DECJ_vals, PMRA_vals, PMDEC_vals, PX_vals],
        names=['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']
    )

    # Set index in your existing DataFrame
    df_indexed = float_df.set_index(['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX'])

    # Reindex to fill missing grid points
    df_full = df_indexed.reindex(full_index).fillna(0.0).reset_index()

    # Optional: Check if it's now complete
    assert df_full.shape[0] == (
            len(RAJ_vals) * len(DECJ_vals) * len(PMRA_vals) * len(PMDEC_vals) * len(PX_vals)
    )

    # Step 2: Sort the DataFrame and reshape the posterior
    df_sorted = df_full.sort_values(['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX'])

    # Reshape into a 5D grid
    posterior_5D = df_sorted['posterior'].values.reshape(
        len(RAJ_vals), len(DECJ_vals), len(PMRA_vals), len(PMDEC_vals), len(PX_vals))

    # Step 3: Nested integration using trapz
    # Integrate over PX (axis=-1)
    int_px = trapz(posterior_5D, x=PX_vals, axis=-1)

    # Integrate over PMDEC (axis=-1 now)
    int_pmdec = trapz(int_px, x=PMDEC_vals, axis=-1)

    # Integrate over PMRA
    int_pmra = trapz(int_pmdec, x=PMRA_vals, axis=-1)

    # Integrate over DECJ
    int_decj = trapz(int_pmra, x=DECJ_vals, axis=-1)

    # Integrate over RAJ (final scalar result)
    integrated_posterior = trapz(int_decj, x=RAJ_vals, axis=-1)

    # Remove the baseline
    posterior_arr = df['posterior'].to_numpy()
    df['posterior'] = np.exp(posterior_arr - np.amax(posterior_arr))

    # Normalize
    df['posterior'] = df['posterior'] / integrated_posterior

    return



def find_timing_label(label):
    if label == 'RAJ':
        return 'ra_t'
    elif label == 'DECJ':
        return 'dec_t'
    elif label == 'PMRA':
        return 'pmra_t'
    elif label == 'PMDEC':
        return 'pmdec_t'
    elif label == 'PX':
        return 'px_t'

def find_timing_error_label(label):
    if label == 'RAJ':
        return 'ra_te'
    elif label == 'DECJ':
        return 'dec_te'
    elif label == 'PMRA':
        return 'pmra_te'
    elif label == 'PMDEC':
        return 'pmdec_te'
    elif label == 'PX':
        return 'px_te'

def label_maker(label):
    if label == 'RAJ':
        return '$\\alpha$'
    elif label == 'DECJ':
        return '$\\delta$'
    elif label == 'PMRA':
        return '$\\mu_\\alpha$'
    elif label == 'PMDEC':
        return '$\\mu_\\delta$'
    elif label == 'PX':
        return '$\\varpi$'

def plot_contour(df, best_sol, timing_astrometric_data, tm, x_label, y_label, ax, colorbars, plot_contours: bool = False):

    all_params = ["RAJ", "DECJ", "PX", "PMRA", "PMDEC"]
    p = [param for param in all_params if param not in (x_label, y_label)]

    sols = df[(df[p[0]] == best_sol[p[0]]) & (df[p[1]] == best_sol[p[1]]) & (df[p[2]] == best_sol[p[2]])]

    x_values = np.sort(sols[x_label].unique())
    y_values = np.sort(sols[y_label].unique())

    print(x_label + ": unique solutions = " + str(len(x_values)))
    print(y_label + ": unique solutions = " + str(len(y_values)))
    print(" ")

    z_values = np.zeros((len(y_values), len(x_values)))

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            try:
                z_values[j, i] = sols[(sols[x_label] == x) & (sols[y_label] == y)]['posterior'].iloc[0]

            # This means that we couldn't find a timing solution for this combination
            except IndexError:
                z_values[j, i] = np.amin(sols['posterior'].to_numpy())
#                z_values[j, i] = np.nan

    ax.grid(True)

    if x_label == 'RAJ':
        timing_RAJ = Angle(timing_astrometric_data['ra_t'], unit=u.hourangle)
        ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 4)}s")
        x = (Angle(x_values, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0
        x_timing = (timing_RAJ - ref_RAJ).hms[2] * 1000.0
        x_timing_error = Angle(timing_astrometric_data['ra_te'], unit=u.hourangle).hms[2] * 1000.0
        best_sol_x = (Angle(best_sol[x_label], unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0
        ax.set_xlabel("$\\alpha- " + f"{ref_RAJ:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
#        ax.set_xticks([-0.003, 0.0, 0.003])
        ax.set_xticks([-0.006, -0.003])
    elif x_label == 'DECJ':
        timing_DECJ = Angle(timing_astrometric_data['dec_t'], unit=u.degree)
        ref_DECJ = Angle(
            f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 3)}s")
        x = (Angle(x_values, unit=u.degree) - ref_DECJ).dms[2] * 1000.0
        x_timing = (timing_DECJ - ref_DECJ).dms[2] * 1000.0
        x_timing_error = Angle(timing_astrometric_data['dec_te'], unit=u.degree).dms[2] * 1000.0
        best_sol_x = (Angle(best_sol[x_label], unit=u.degree) - ref_DECJ).dms[2] * 1000.0
        if ref_DECJ.dms[0] > 0:
            ax.set_xlabel("$\delta - " + f"{ref_DECJ:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
        else:
            ax.set_xlabel("$\delta + " + f"{ref_DECJ:latex}"[2:-1] + "$\n$[\mathrm{mas}]$")
        ax.set_xticks([0.2, 0.4, 0.6])
    else:
        x = x_values
        x_timing = timing_astrometric_data[find_timing_label(x_label)],
        x_timing_error = timing_astrometric_data[find_timing_error_label(x_label)]
        best_sol_x = best_sol[x_label]
        ax.set_xlabel(f"{label_maker(x_label)}\n[{getattr(tm, x_label).units}]")

    if y_label == 'RAJ':
        timing_RAJ = Angle(timing_astrometric_data['ra_t'], unit=u.hourangle)
        ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 1)}s")
        y = (Angle(y_values, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0
        y_timing = (timing_RAJ - ref_RAJ).hms[2] * 1000.0
        y_timing_error = Angle(timing_astrometric_data['ra_te'], unit=u.hourangle).hms[2] * 1000.0
        best_sol_y = (Angle(best_sol[y_label], unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0
        ax.set_ylabel("$\\alpha - " + f"{ref_RAJ:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
    elif y_label == 'DECJ':
        timing_DECJ = Angle(timing_astrometric_data['dec_t'], unit=u.degree)
        ref_DECJ = Angle(
            f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 3)}s")
        y = (Angle(y_values, unit=u.degree) - ref_DECJ).dms[2] * 1000.0
        y_timing = (timing_DECJ - ref_DECJ).dms[2] * 1000.0
        y_timing_error = Angle(timing_astrometric_data['dec_te'], unit=u.degree).dms[2] * 1000.0
        best_sol_y = (Angle(best_sol[y_label], unit=u.degree) - ref_DECJ).dms[2] * 1000.0
        if ref_DECJ.dms[0] > 0:
            ax.set_ylabel("$\delta - " + f"{ref_DECJ:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
        else:
            ax.set_ylabel("$\delta + " + f"{ref_DECJ:latex}"[2:-1] + "$\n$[\mathrm{mas}]$")
    else:
        y = y_values
        y_timing = timing_astrometric_data[find_timing_label(y_label)]
        y_timing_error = timing_astrometric_data[find_timing_error_label(y_label)]
        best_sol_y = best_sol[y_label]
        ax.set_ylabel(f"{label_maker(y_label)}\n[{getattr(tm, y_label).units}]")

    # Plot contour
    if plot_contours:
        contour = ax.contourf(x, y, z_values, levels=20, cmap="viridis")
        colorbars.append(contour)
        #    cbar = plt.colorbar(contour, ax=ax, label='posterior')
        #    cbar.remove()  # Remove it
    else:
        cmap = ax.pcolormesh(x, y, z_values/np.nanmax(z_values), cmap="viridis")
        colorbars.append(cmap)

    if PSR_name == "J0030+0451" or PSR_name == "J2145-0750":
        ax.patch.set_facecolor(plt.cm.viridis(0))
        ax.grid(False)

    ax.axvline(x=best_sol_x, color='k', linestyle='--', linewidth=2.5)
    ax.axhline(y=best_sol_y, color='k', linestyle='--', linewidth=2.5)

    # Extract the reference timing values
    ax.scatter(x=x_timing, y=y_timing, marker='x', c='red', s=400)
    ax.errorbar(x=x_timing, y=y_timing, xerr=x_timing_error, yerr=y_timing_error, marker='x', c='red', elinewidth=4, capthick=4, capsize=12)

#    ax.set_title(f'{x_col} vs {y_col} with {w_col} as color')

    return


if __name__ == "__main__":

#    PSR_name: str = "J0030+0451"
#    PSR_name: str = "J1730-2304"
#    PSR_name: str = "J1640+2224"
#    PSR_name: str = "J1918-0642"
#    PSR_name: str = "J2010-1323"
#    PSR_name: str = "J2145-0750"
    PSR_name: str = "J2317+1439"
    posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"
    float_posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors_floats.pkl"

    # Get the nominal timing values
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    tm = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    # Load the timing solution
    timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0).loc[PSR_name]
    print(timing_astrometric_data)

    # Load the posteriors
    result_df = pd.read_pickle(posteriors_file)
    print(f"Number of solutions = {len(result_df.index)}")

    # Convert PX, PMRA, PMDEC to float
    result_df[["PX", "PMRA", "PMDEC", "posterior"]] = result_df[["PX", "PMRA", "PMDEC", "posterior"]].astype(float)

    # De-log the posteriors
    delog(result_df)

    # Find the solution with the highest posterior
    best_sol, best_sol_err = find_best_sol(result_df)
    print(f"Best solution = {best_sol}")
    print(f"Best solution error = {best_sol_err}")

    # Create subplots
    sns.set_context('paper')
    sns.set_style('ticks')
    sns.set(font_scale=4.5)
    fig, axs = plt.subplots(4, 4, figsize=(36, 30), gridspec_kw = {'wspace':0.1, 'hspace':0.1})
    fig.tight_layout()
#    fig.suptitle(PSR_name)

    # Store contour plots for color normalization
    colorbars = []

    for row in range(4):

        for col in range(row+1, 4):
            axs[row, col].axis('off')

        for col in range(4):
            if col > 0:
                axs[row, col].get_yaxis().set_visible(False)
            if row < 3:
                axs[row, col].get_xaxis().set_visible(False)

    # Plot each pair
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'DECJ', axs[0, 0], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'PMRA', axs[1, 0], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'DECJ', 'PMRA', axs[1, 1], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'PMDEC', axs[2, 0], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'DECJ', 'PMDEC', axs[2, 1], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'PMRA', 'PMDEC', axs[2, 2], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'RAJ', 'PX', axs[3, 0], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'DECJ', 'PX', axs[3, 1], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'PMRA', 'PX', axs[3, 2], colorbars)
    plot_contour(result_df, best_sol, timing_astrometric_data, tm, 'PMDEC', 'PX', axs[3, 3], colorbars)

    # Synchronize x-limits in each column

    for col in range(4):
        x_min = min(ax.get_xlim()[0] for ax in axs[col:, col])
        x_max = max(ax.get_xlim()[1] for ax in axs[col:, col])
        for ax in axs[:, col]:
            ax.set_xlim(x_min, x_max)

    # Synchronize y-limits in each row
    for row in range(4):
        y_min = min(ax.get_ylim()[0] for ax in axs[row, :row+1])
        y_max = max(ax.get_ylim()[1] for ax in axs[row, :row+1])
        for ax in axs[row, :]:
            ax.set_ylim(y_min, y_max)

    plt.savefig("./figures/corner_plot_" + PSR_name + "_nolog.pdf")
    plt.show()