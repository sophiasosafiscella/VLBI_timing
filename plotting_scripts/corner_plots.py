#----------------------------------------------------------------------------------------------------------------------
# Make posterior corner plots
#----------------------------------------------------------------------------------------------------------------------

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from matplotlib import ticker
from pint.models import get_model
import glob


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

def plot_contour(df, x_label, y_label, ax):

    # pivot: marginalize (sum) over all other parameters
    grid = df.pivot_table(index=y_label, columns=x_label, values='posterior', aggfunc='sum', fill_value=0.0)

    # Sort rows and columns so the edges are correct
    grid = grid.sort_index(axis=0)
    grid = grid.sort_index(axis=1)

    H = grid.values
    H = H / H.sum()  # normalize to 1

    x_centers = grid.columns.values
    y_centers = grid.index.values

    # compute edges
    if len(x_centers) > 1:
        dx = np.diff(x_centers).min()
        x_edges = np.concatenate(
            [[x_centers[0] - dx / 2], x_centers[:-1] + np.diff(x_centers) / 2, [x_centers[-1] + dx / 2]])
    else:
        x_edges = [x_centers[0] - 0.5, x_centers[0] + 0.5]

    if len(y_centers) > 1:
        dy = np.diff(y_centers).min()
        y_edges = np.concatenate(
            [[y_centers[0] - dy / 2], y_centers[:-1] + np.diff(y_centers) / 2, [y_centers[-1] + dy / 2]])
    else:
        y_edges = [y_centers[0] - 0.5, y_centers[0] + 0.5]

    X, Y = np.meshgrid(x_edges, y_edges)

    # plot
    if PSR_name == "J0030+0451" or PSR_name == "J2145-0750":
        ax.patch.set_facecolor(plt.cm.viridis(0))
    cmap = ax.pcolormesh(X, Y, H, cmap="viridis", edgecolors='grey', linewidth=0.1)
    ax.grid(True)

    if x_label == 'RAJ':
        x_timing = timing_astrometric_data['ra_t']
        x_timing_error = timing_astrometric_data['ra_te']
        ax.set_xlabel("$\\alpha- " + f"{timing_astrometric_data['ref_RAJ']:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
    #        ax.set_xticks([-0.003, 0.0, 0.003])
    #        ax.set_xticks([-0.006, -0.003])
    elif x_label == 'DECJ':
        x_timing = timing_astrometric_data['dec_t']
        x_timing_error = timing_astrometric_data['dec_te']
        ref_DECJ = timing_astrometric_data['ref_DECJ']
        if ref_DECJ.dms[0] > 0:
            ax.set_xlabel("$\delta - " + f"{ref_DECJ:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
        else:
            ax.set_xlabel("$\delta + " + f"{ref_DECJ:latex}"[2:-1] + "$\n$[\mathrm{mas}]$")
    #        ax.set_xticks([0.2, 0.4, 0.6])
    else:
        x_timing = timing_astrometric_data[find_timing_label(x_label)]
        x_timing_error = timing_astrometric_data[find_timing_error_label(x_label)]
        ax.set_xlabel(f"{label_maker(x_label)}\n[{getattr(tm, x_label).units}]")

    if y_label == 'RAJ':
        y_timing = timing_astrometric_data['ra_t']
        y_timing_error = timing_astrometric_data['ra_te']
        ax.set_ylabel("$\\alpha - " + f"{timing_astrometric_data['ref_RAJ']:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
    elif y_label == 'DECJ':
        y_timing = timing_astrometric_data['dec_t']
        y_timing_error = timing_astrometric_data['dec_te']
        ref_DECJ = timing_astrometric_data['ref_DECJ']
        if ref_DECJ.dms[0] > 0:
            ax.set_ylabel("$\delta - " + f"{ref_DECJ:latex}"[1:-1] + "$\n$[\mathrm{mas}]$")
        else:
            ax.set_ylabel("$\delta + " + f"{ref_DECJ:latex}"[2:-1] + "$\n$[\mathrm{mas}]$")
    else:
        y_timing = timing_astrometric_data[find_timing_label(y_label)]
        y_timing_error = timing_astrometric_data[find_timing_error_label(y_label)]
        ax.set_ylabel(f"{label_maker(y_label)}\n[{getattr(tm, y_label).units}]")

    # Extract the reference timing values
    ax.scatter(x=x_timing, y=y_timing, marker='x', c='red', s=400)
    ax.errorbar(x=x_timing, y=y_timing, xerr=x_timing_error, yerr=y_timing_error, marker='x', c='red', elinewidth=4,
                capthick=4, capsize=12)

    return


if __name__ == "__main__":

    #PSR_name: str = "J0030+0451"
    #PSR_name: str = "J1730-2304"
    #PSR_name: str = "J1640+2224"
    #PSR_name: str = "J1918-0642"
    #PSR_name: str = "J2010-1323"
    #PSR_name: str = "J2145-0750"
    PSR_name: str = "J2317+1439"
    posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors.pkl"
    float_posteriors_file: str = f"./results/timing_posteriors_frame_tie/{PSR_name}_consolidated_timing_posteriors_floats.pkl"

    # Get the nominal timing values
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    tm = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    # Load the timing solution
    timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0).loc[PSR_name]

    # Load the posteriors
    result_df = pd.read_pickle(posteriors_file)
    print(f"Number of solutions = {len(result_df.index)}")

    # Convert PX, PMRA, PMDEC to float
    result_df[["PX", "PMRA", "PMDEC", "posterior"]] = result_df[["PX", "PMRA", "PMDEC", "posterior"]].astype(float)

    # For numerical stability
    log_posteriors = result_df['posterior'].values
    log_posteriors -= np.max(log_posteriors)

    # exponentiate to get probabilities (linear scale)
    posteriors = np.exp(log_posteriors)

    # normalize
    posteriors /= posteriors.sum()
    result_df['posterior'] = posteriors

    # Convert the RA
    timing_RAJ = Angle(timing_astrometric_data['ra_t'], unit=u.hourangle)
    ref_RAJ = Angle(f"{int(timing_RAJ.hms[0])}h{int(timing_RAJ.hms[1])}m{round(timing_RAJ.hms[2], 4)}s")
    result_df['RAJ'] = ((Angle(result_df['RAJ'].values, unit=u.hourangle) - ref_RAJ).hms[2] * 1000.0)  # Milliseconds of time
    timing_astrometric_data['ref_RAJ'] = ref_RAJ
    timing_astrometric_data['ra_t'] = ((timing_RAJ - ref_RAJ).hms[2] * 1000.0)
    timing_astrometric_data['ra_te'] = (Angle(timing_astrometric_data['ra_te'], unit=u.hourangle).hms[2] * 1000.0)

    # Convert the DEC
    timing_DECJ = Angle(timing_astrometric_data['dec_t'], unit=u.degree)
    ref_DECJ = Angle(f"{int(timing_DECJ.dms[0])}d{int(abs(timing_DECJ.dms[1]))}m{round(abs(timing_DECJ.dms[2]), 3)}s")
    result_df['DECJ'] = (Angle(result_df['DECJ'].values, unit=u.deg) - ref_DECJ).to(u.mas).value
    timing_astrometric_data['ref_DECJ'] = ref_DECJ
    timing_astrometric_data['dec_t'] = (timing_DECJ - ref_DECJ).to(u.mas).value
    timing_astrometric_data['dec_te'] = Angle(timing_astrometric_data['dec_te'], unit=u.degree).to(u.mas).value

    # Create subplots
    sns.set_context('paper')
    sns.set_style('ticks')
    sns.set(font_scale=4.5)
    fig, axs = plt.subplots(4, 4, figsize=(36, 30), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for row in range(4):

        for col in range(row+1, 4):
            axs[row, col].axis('off')

        for col in range(4):
            if col > 0:
                axs[row, col].get_yaxis().set_visible(False)
            if row < 3:
                axs[row, col].get_xaxis().set_visible(False)

    # Store contour plots for color normalization
    colorbars = []

    # Plot each pair
    plot_contour(result_df,'RAJ', 'DECJ', axs[0, 0])
    plot_contour(result_df,'RAJ', 'PMRA', axs[1, 0])
    plot_contour(result_df,'DECJ', 'PMRA', axs[1, 1])
    plot_contour(result_df,'RAJ', 'PMDEC', axs[2, 0])
    plot_contour(result_df,'DECJ', 'PMDEC', axs[2, 1])
    plot_contour(result_df,'PMRA', 'PMDEC', axs[2, 2])
    plot_contour(result_df,'RAJ', 'PX', axs[3, 0])
    plot_contour(result_df,'DECJ', 'PX', axs[3, 1])
    plot_contour(result_df,'PMRA', 'PX', axs[3, 2])
    plot_contour(result_df,'PMDEC', 'PX', axs[3, 3])

    # Synchronize x-limits in each column
    for col in range(4):
        # Only check the limits of the plots in the lower triangle (row >= col)
        plotted_axes = [axs[row, col] for row in range(col, 4)]
        x_min = min(ax.get_xlim()[0] for ax in plotted_axes)
        x_max = max(ax.get_xlim()[1] for ax in plotted_axes)

        # Apply the new limits only to those same plots
        for ax in plotted_axes:
            ax.set_xlim(x_min, x_max)

    # Synchronize y-limits in each row
    for row in range(4):
        # Only check the limits of the plots in the lower triangle (col <= row)
        plotted_axes = [axs[row, col] for col in range(row + 1)]
        y_min = min(ax.get_ylim()[0] for ax in plotted_axes)
        y_max = max(ax.get_ylim()[1] for ax in plotted_axes)

        # Apply the new limits only to those same plots
        for ax in plotted_axes:
            ax.set_ylim(y_min, y_max)

    plt.savefig("./figures/corner_plot_" + PSR_name + "_marginalized.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

