import sys
from itertools import product

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from astropy.coordinates import Angle
from matplotlib.patches import Patch
from shapely import prepare
from shapely.geometry import Polygon
from uncertainties import ufloat, umath

from VLBI_utils import pdf_values


def circle(radius, center):
    angle = np.linspace(0, 2 * np.pi, 150)
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]

    return np.column_stack((x, y))


def overlap_range(x, y, factor, grid_num):
    # Calculate the bounds for both distributions
    lower1, upper1 = x.nominal_value - factor * x.std_dev, x.nominal_value + factor * x.std_dev
    lower2, upper2 = y.nominal_value - factor * y.std_dev, y.nominal_value + factor * y.std_dev

    # Calculate the overlap range
    lower_overlap = max(lower1, lower2)
    upper_overlap = min(upper1, upper2)

    # Check if there is an overlap
    if lower_overlap < upper_overlap:
        return (lower_overlap, upper_overlap), np.linspace(lower_overlap, upper_overlap, grid_num)
    else:
        return None, None # No overlap

def asymmetrical_overlap_range(x, y, y_uL, y_uR, factor, grid_num):
    # Calculate the bounds for both distributions
    lower1, upper1 = x.nominal_value - factor * x.std_dev, x.nominal_value + factor * x.std_dev
    lower2, upper2 = y - factor * y_uL, y + factor * y_uR

    # Calculate the overlap range
    lower_overlap = max(lower1, lower2)
    upper_overlap = min(upper1, upper2)

    # Check if there is an overlap
    if lower_overlap < upper_overlap:
        return (lower_overlap, upper_overlap), np.linspace(lower_overlap, upper_overlap, grid_num)
    else:
        return None, None # No overlap


def pm_overlap_range(timing_PMRA,timing_PMDEC, mu_PM_VLBI, uL_PM_VLBI, uR_PM_VLBI, n=3, grid_num=100):
    # Calculate the bounds for both distributions
    lower_PMRA_timing, upper_PMRA_timing = timing_PMRA.nominal_value - n * timing_PMRA.std_dev, timing_PMRA.nominal_value + n * timing_PMRA.std_dev
    lower_PMDEC_timing, upper_PMDEC_timing = timing_PMDEC.nominal_value - n * timing_PMDEC.std_dev, timing_PMDEC.nominal_value + n * timing_PMDEC.std_dev
    lower_PM_VLBI, upper_PM_VLBI = mu_PM_VLBI - n * uL_PM_VLBI, mu_PM_VLBI + n * uR_PM_VLBI

    pmra_list = np.linspace(lower_PMRA_timing, upper_PMRA_timing, grid_num)
    pmdec_list = np.linspace(lower_PMDEC_timing, upper_PMDEC_timing, grid_num)

    overlaps = [(pmra, pmdec) for pmra, pmdec in product(pmra_list, pmdec_list) if lower_PM_VLBI ** 2 <= pmra ** 2 + pmdec ** 2 <= upper_PM_VLBI ** 2]

    if len(overlaps) > 0:
        return True, overlaps
    else:
        return None, None


def plot_overlap(param, overlap, timing, VLBI, fig, row, col):

    VLBI_color = "rgba(0, 204, 150, 0.5)"  # px.colors.qualitative.Pastel1[2]
    timing_color = "rgba(99, 110, 250, 0.5)"  # px.colors.qualitative.Pastel1[1]

    fig.add_trace(go.Scatter(x=overlap, y=np.full(len(overlap), 1.0), mode='lines+markers', marker=dict(color='red')),
                  row=row, col=col)

    fig.add_vline(x=timing.nominal_value, line_width=3, line_dash="dash", line_color=timing_color, row=row, col=col)
    fig.add_vrect(x0=timing.nominal_value - 3 * timing.std_dev, x1=timing.nominal_value + 3 * timing.std_dev, line_width=0,
                  fillcolor=timing_color, opacity=0.3, row=row, col=col)

    fig.add_vline(x=VLBI.nominal_value, line_width=3, line_dash="dash", line_color=VLBI_color, row=row, col=col)
    fig.add_vrect(x0=VLBI.nominal_value - 3 * VLBI.std_dev, x1=VLBI.nominal_value + 3 * VLBI.std_dev, line_width=0,
                  fillcolor=VLBI_color, opacity=0.3, row=row, col=col)

    fig.update_yaxes(showticklabels=False, row=row, col=col)
    fig.update_xaxes(title_text=param, row=row, col=col)

    return


def find_solutions(PSR_name, VLBI_data, timing_data, factor: int = 3, grid_num: int = 10, plot=False):
    VLBI_color = (0 / 255, 204 / 255, 150 / 255, 0.5)  # equivalent to rgba(0, 204, 150, 0.5)
    timing_color = (99 / 255, 110 / 255, 250 / 255, 0.5)  # equivalent to rgba(99, 110, 250, 0.5)

    #    fig = make_subplots(rows=1, cols=4)

    # ------------------------------RAJ------------------------------
    timing_RAJ = ufloat(Angle(timing_data.loc[PSR_name, "ra_t"]).rad, Angle(timing_data.loc[PSR_name, "ra_te"]).rad)
    VLBI_RAJ = ufloat(Angle(VLBI_data.loc[PSR_name, "ra_v"]).rad, Angle(VLBI_data.loc[PSR_name, "ra_ve"]).rad)

    RAJ_overlap, RAJ_values = overlap_range(timing_RAJ, VLBI_RAJ, factor, grid_num)

#    plot_overlap("RAJ", RAJ_values, timing_RAJ, VLBI_RAJ, fig, 1, 1)

    # ------------------------------DECJ------------------------------
    timing_DECJ = ufloat(Angle(timing_data.loc[PSR_name, 'dec_t']).rad, Angle(timing_data.loc[PSR_name, "dec_te"]).rad)
    VLBI_DECJ = ufloat(Angle(VLBI_data.loc[PSR_name, "dec_v"]).rad, Angle(VLBI_data.loc[PSR_name, "dec_ve"]).rad)

    DECJ_overlap, DECJ_values = overlap_range(timing_DECJ, VLBI_DECJ, factor, grid_num)

#    plot_overlap("DECJ", DECJ_values, timing_DECJ, VLBI_DECJ, fig, 1, 2)

    '''
    # ------------------------------PMRA------------------------------
    PMRA_overlap, PMRA_values = asymmetrical_overlap_range(eq_timing_model.PMRA.value,
                                                           eq_timing_model.PMRA.uncertainty.value,
                                                           eq_timing_model.PMRA.uncertainty.value,
                                                           data.loc[PSR_name, "VLBI_PMRA"],
                                                           data.loc[PSR_name, "VLBI_PMRA_uL"],
                                                           data.loc[PSR_name, "VLBI_PMRA_uR"], factor, grid_num)

    # ------------------------------PMDEC------------------------------
    PMDEC_overlap, PMDEC_values = asymmetrical_overlap_range(eq_timing_model.PMDEC.value,
                                                           eq_timing_model.PMDEC.uncertainty.value,
                                                           eq_timing_model.PMDEC.uncertainty.value,
                                                           data.loc[PSR_name, "VLBI_PMDEC"],
                                                           data.loc[PSR_name, "VLBI_PMDEC_uL"],
                                                           data.loc[PSR_name, "VLBI_PMDEC_uR"], factor, grid_num)
    '''

    # ------------------------------Proper Motion------------------------------
    # Timing
    timing_PMRA = ufloat(timing_data.loc[PSR_name, "pmra_t"], timing_data.loc[PSR_name, "pmra_te"])
    timing_PMDEC = ufloat(timing_data.loc[PSR_name, "pmdec_t"], timing_data.loc[PSR_name, "pmdec_te"])

    # VLBI: sometimes the error bars are asymmetric. We will propagate errors twice, each time assuming a symmetric
    # error equal to either VLBI_uL or VLBI_uR, and then keep the standard deviations of each sum:
    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(VLBI_data.loc[PSR_name, "pmra_v"], VLBI_data.loc[PSR_name, "pmra_v_" + error_side])
        VLBI_PMDEC = ufloat(VLBI_data.loc[PSR_name, "pmdec_v"], VLBI_data.loc[PSR_name, "pmdec_v_" + error_side])
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2)

        if error_side == "uL":
            VLBI_PM_uL = VLBI_PM.std_dev
        elif error_side == "uR":
            VLBI_PM_uR = VLBI_PM.std_dev

    # Find the overlaps
    PM_overlap, PM_values = pm_overlap_range(timing_PMRA, timing_PMDEC, VLBI_PM, VLBI_PM_uL, VLBI_PM_uR, factor, grid_num)

    if RAJ_overlap and DECJ_overlap and plot:
        timing_PM = umath.sqrt(timing_PMRA ** 2 + timing_PMDEC ** 2)

        # Make a grid of PMRA/PMDEC values, and calculate probability density functions (PDFs) for each of those values
        PMRA_values, PMRA_pdf = pdf_values(x0=timing_PMRA.nominal_value, uL=timing_PMRA.std_dev, uR=timing_PMRA.std_dev,
                                           factor=factor, num=grid_num)
        PMDEC_values, PMDEC_pdf = pdf_values(x0=timing_PMDEC.nominal_value, uL=timing_PMDEC.std_dev, uR=timing_PMDEC.std_dev,
                                             factor=factor, num=grid_num)

        # Calculate the joint probability distribution by multiplying the PDFs
        joint_pdf = np.outer(PMRA_pdf, PMDEC_pdf)

        # Normalize the joint PDF so that it sums to 1
        joint_pdf /= np.sum(joint_pdf)

        # Create a meshgrid for contour plotting
        X, Y = np.meshgrid(PMRA_values, PMDEC_values)

        # Create figure and axes objects for main plot and marginal distributions
        sns.set_theme(context="paper", style="ticks", font_scale=2.0,
                      rc={"axes.axisbelow": False, "grid.linewidth": 1.4})
        fig, ([ax_marginal_x, other], [ax_main, ax_marginal_y]) = plt.subplots(2, 2, figsize=(9, 7),
                                                                               gridspec_kw={'height_ratios': [1, 4],
                                                                                            'width_ratios': [4, 1],
                                                                                            'hspace': 0.00,
                                                                                            'wspace': 0.00})

        # Plot contour plot on main axes
        contour = ax_main.contourf(X, Y, joint_pdf, cmap="viridis", zorder=0)
        cbar = plt.colorbar(contour, ax=ax_main, label='PDF', location="left", pad=-0.15,
                            anchor=(-2.5, 0.5))

        # Format the colorbar ticks to show mantissa and exponent separately
        def fmt(x, p):
            if x == 0:
                return '0'
            exp = int(np.floor(np.log10(abs(x))))
            mantissa = x / (10 ** exp)
            return f'{mantissa:.1f}'

        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
        # Add exponent text at the top of the colorbar
        max_value = np.max(joint_pdf)
        exp = int(np.floor(np.log10(abs(max_value))))
        cbar.ax.text(0.5, 1.05, f'×10$^{{{exp}}}$', ha='center', va='bottom', transform=cbar.ax.transAxes)
        ax_main.set_xlabel("$\mu_{\\alpha^{*}} = \mu_{\\alpha} \cos(\delta)$", fontsize=18)
        ax_main.set_ylabel("$\mu_{\delta}$", fontsize=18)

        # Create a polygon object containing the timing solution
        timing_polygon = Polygon(contour.allsegs[1][0])
        prepare(timing_polygon)

        # Plot marginal distribution for X on top right subplot
        ax_marginal_x.plot(PMRA_values, PMRA_pdf, color='red')
        ax_marginal_x.set_xlim(ax_main.get_xlim())
        ax_marginal_x.set_ylim(0, np.max(PMRA_pdf) * 1.1)
        ax_marginal_x.set_xticks([])
        ax_marginal_x.tick_params(left=False)  # remove the ticks
        ax_marginal_x.set(yticklabels=[])

        #    ax_marginal_x.set_title('PDF of X')

        # Plot marginal distribution for Y on bottom left subplot
        ax_marginal_y.plot(PMDEC_pdf, PMDEC_values, color='blue')
        ax_marginal_y.set_ylim(ax_main.get_ylim())
        ax_marginal_y.set_xlim(0, np.max(PMDEC_pdf) * 1.1)
        ax_marginal_y.set_yticks([])
        ax_marginal_y.tick_params(bottom=False)  # remove the ticks
        ax_marginal_y.set(xticklabels=[])
        #    ax_marginal_y.set_title('PDF of Y')

        # Remove unnecessary spines
        other.spines['right'].set_visible(False)
        other.spines['top'].set_visible(False)
        other.set_xticks([])
        other.set_yticks([])

        # Create a circle containing the +/- values
        circle_timing_out = plt.Circle((0, 0), radius=timing_PM.nominal_value + factor * timing_PM.std_dev, color='r',
                                       lw=3, ls=":",
                                       fill=False,
                                       label="$\mu_{\mathrm{timing}} \pm$" + str(factor) + "$\sigma_{\mu}$")
        circle_timing_in = plt.Circle((0, 0), radius=timing_PM.nominal_value - factor * timing_PM.std_dev, color='r',
                                      lw=3, ls=":",
                                      fill=False)
        ax_main.add_patch(circle_timing_out)
        ax_main.add_patch(circle_timing_in)

        # Make circles representing the PM +/- sigma region for VLBI
        circle_uL = plt.Circle((0, 0), VLBI_PM.nominal_value - factor * VLBI_PM.std_dev, color='g', lw=3, ls="--",
                               fill=False,
                               label="$\mu_{\mathrm{VLBI}} \pm$" + str(factor) + "$\sigma_{\mu}$")
        circle_ML = plt.Circle((0, 0), VLBI_PM.nominal_value, color='b', lw=4, fill=False,
                               label="$\mu_{\mathrm{VLBI}}$")
        circle_uR = plt.Circle((0, 0), VLBI_PM.nominal_value + factor * VLBI_PM.std_dev, color='g', lw=3, ls="--",
                               fill=False)
        ax_main.add_patch(circle_uL)
        ax_main.add_patch(circle_ML)
        ax_main.add_patch(circle_uR)
        ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.savefig("./results/frame_tie/" + PSR_name + "_PM.pdf", bbox_inches='tight')
        plt.show()
        sys.exit()
#    plot_overlap("PM", DECJ_values, timing_DECJ, VLBI_DECJ, fig, 1, 2)

    # ------------------------------Parallax------------------------------
    timing_PX = ufloat(timing_data.loc[PSR_name, "px_t"], timing_data.loc[PSR_name, "px_te"])

    PX_overlap, PX_values = asymmetrical_overlap_range(timing_PX, VLBI_data.loc[PSR_name, "px_v"],
                                                       VLBI_data.loc[PSR_name, "px_v_uL"], VLBI_data.loc[PSR_name, "px_v_uR"],
                                                       factor, grid_num)

    if RAJ_overlap and DECJ_overlap and plot:
#        fig = go.Figure()
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        sns.set_theme(context="paper", style="ticks", font_scale=2)

        PX_values, PX_pdf = pdf_values(x0=timing_PX.nominal_value, uL=timing_PX.std_dev, uR=timing_PX.std_dev, factor=factor, num=grid_num)

        VLBI_PX_x0: float = VLBI_data.loc[PSR_name, "px_v"]
        VLBI_PX_uL: float = VLBI_data.loc[PSR_name, "px_v_uL"]
        VLBI_PX_uR: float = VLBI_data.loc[PSR_name, "px_v_uR"]

#        ax.fill_between(x=PX_values, y=PX_pdf, name="Timing", fill='tozeroy', fillcolor=timing_color, mode='none'))
        ax.plot(PX_values, PX_pdf, color=timing_color, zorder=10)
        ax.fill_between(PX_values, PX_pdf, color=timing_color, zorder=10, label="Timing")
        ax.axvline(x=timing_PX.nominal_value - 3 * timing_PX.std_dev, lw=3, ls='--', alpha=1, c=timing_color)
        ax.axvline(x=timing_PX.nominal_value + 3 * timing_PX.std_dev, lw=3, ls='--', alpha=1, c=timing_color)
        ax.text(0.06, 0.6, "$\\varpi^\mathrm{t} \pm 3\sigma_{\\varpi^\mathrm{t}}$", transform=ax.transAxes, fontsize=18,
        verticalalignment='top', c=timing_color, alpha=1)

        x, y = pdf_values(x0=VLBI_PX_x0, uL=VLBI_PX_uL, uR=VLBI_PX_uR, factor=factor, num=grid_num)
#        fig.add_trace(go.Scatter(x=x, y=y, name="VLBI", fill='tozeroy', fillcolor=VLBI_color, mode='none'))
        ax.plot(x, y, color=VLBI_color, zorder=10)
        ax.fill_between(x, y, color=VLBI_color, zorder=10, label="VLBI")
        ax.axvline(x=VLBI_PX_x0 - factor * VLBI_PX_uL, lw=3, ls='--', alpha=1, c=VLBI_color)
        ax.axvline(x=VLBI_PX_x0 + factor * VLBI_PX_uR, lw=3, ls='--', alpha=1, c=VLBI_color)
        ax.text(0.69, 0.4, "$\\varpi^\mathrm{VLBI} + 3u_\mathrm{R}$", transform=ax.transAxes, fontsize=18,
        verticalalignment='top', c=VLBI_color, alpha=1)
        ax.text(0.23, 0.4, "$\\varpi^\mathrm{VLBI} - 3u_\mathrm{L}$", transform=ax.transAxes, fontsize=18,
        verticalalignment='top', c=VLBI_color, alpha=1)

        # Overlap
        ax.axvspan(VLBI_PX_x0 - factor * VLBI_PX_uL, VLBI_PX_x0 + factor * VLBI_PX_uR, facecolor='none', edgecolor='gray', hatch='//', linewidth=0.0, zorder=0, label="Overlap")
        ax.tick_params(left=False)  # remove the ticks
        ax.set(yticklabels=[])
        hatch_patch = Patch(facecolor='none', edgecolor='gray', hatch='//', label='Overlap region')
        ax.legend(handles=[hatch_patch])


        ax.set_xlabel("$\\varpi [\mathrm{mas}]$")
        ax.set_ylabel("PDF")
        ax.set_ylim([0, ax.get_ylim()[-1]])
        plt.legend()
        plt.grid(zorder=0)
        plt.tight_layout()
        fig.savefig(f"./results/frame_tie/{PSR_name[0:5]}_PX.pdf", bbox_inches='tight')
        fig.show()
    # ------------------------------Find the overlap------------------------------

    if RAJ_overlap and DECJ_overlap and PM_overlap and PX_overlap:
        print("Overlaps found!")
        RAJ_values_hms = Angle(RAJ_values, unit=u.rad).to_string(unit=u.hourangle, sep=':')
        DECJ_values_dms = Angle(DECJ_values, unit=u.rad).to_string(unit=u.degree, sep=':')
        return product(RAJ_values_hms, DECJ_values_dms, PX_values, PM_values)

    else:
        if not RAJ_overlap:
            print("There is no overlap in RAJ")
        if not DECJ_overlap:
            print("There is no overlap in DECJ")
        if not PM_overlap:
            print("There is no overlap in PM")
        if not PX_overlap:
            print("There is no overlap in PX")
        return None


if __name__ == "__main__":

    PSR_name: str = sys.argv[1]

    # File containing the timing and VLBI astrometric VLBI_data
    VLBI_astrometric_data = pd.read_csv("./data/frame_tied_vlbi_astrometric_data.csv", index_col=0, header=0)
    timing_astrometric_data = pd.read_csv("./data/timing_astrometric_data_updated.csv", index_col=0, header=0)
    PSR_list = VLBI_astrometric_data.index  # List of pulsars

    print(f"Finding the possible timing solutions for {PSR_name}")

    # FIND THE OVERLAP BETWEEN THE TIMING AND V LBI SOLUTIONS
    solutions = find_solutions(PSR_name, VLBI_astrometric_data, timing_astrometric_data, grid_num=100, plot=True)

    if solutions:
        overlap_df = pd.DataFrame(data=solutions, columns=["RA", "DEC", "PX", "PM"])
        overlap_df[['PMRA', 'PMDEC']] = pd.DataFrame(overlap_df['PM'].tolist(), index=overlap_df.index)
        overlap_df = overlap_df.drop(columns=['PM'])
        overlap_df['POSEPOCH'] = timing_astrometric_data.loc[PSR_name, "epoch_t"]
#            overlap_df.to_pickle(f"./results/frame_tie/{PSR_name}_overlap_frame_tie.pkl")
#        overlap_df.to_csv(f"./results/frame_tie/{PSR_name}_overlap_frame_tie.txt", sep=" ", header=True, index_label="ArrayTaskID")
