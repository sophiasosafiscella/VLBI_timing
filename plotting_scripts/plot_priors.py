#----------------------------------------------------------------------------------------------------------------------
# Plot an example of prior calculation
#----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from VLBI_utils import pdf_values
from uncertainties import ufloat
from astropy.coordinates import Angle
import astropy.units as u
import sys

PSR_name: str = "J2145-0750"
grid_num: int = 50
factor: int = 2.8


VLBI_data = pd.read_csv("./data/vlbi_astrometric_data_frame_tied.csv", index_col=0, header=0)

RAJ, RAJ_err = Angle(VLBI_data.loc[PSR_name, "ra_v"]), Angle(VLBI_data.loc[PSR_name, "ra_ve"])
ref_RAJ = Angle(f"{int(RAJ.hms[0])}h{int(RAJ.hms[1])}m{round(RAJ.hms[2], 1)}s")
VLBI_deltaRAJ_ms = (RAJ.to(u.hourangle) - ref_RAJ).hms[2] * 1000.0
VLBI_RAJ_err_ms = RAJ_err.hms[2] * 1000.0
RAJ_values, RAJ_pdf = pdf_values(x0=VLBI_deltaRAJ_ms, uL=VLBI_RAJ_err_ms, uR=VLBI_RAJ_err_ms, factor=factor)

trial_RAJ_ms = (Angle("21h45m50.4587s") - ref_RAJ).hms[2] * 1000.0

DECJ, DECJ_err = Angle(VLBI_data.loc[PSR_name, "dec_v"]), Angle(VLBI_data.loc[PSR_name, "dec_ve"])
ref_DECJ = Angle(f"{int(DECJ.dms[0])}d{int(abs(DECJ.dms[1]))}m{round(abs(DECJ.dms[2]),2)}s")
VLBI_deltaDECJ_ms = (DECJ.to(u.hourangle) - ref_DECJ).dms[2] * 1000.0
VLBI_DECJ_err_ms = DECJ_err.dms[2] * 1000.0
DECJ_values, DECJ_pdf = pdf_values(x0=VLBI_deltaDECJ_ms, uL=VLBI_DECJ_err_ms, uR=VLBI_DECJ_err_ms, factor=factor)

PX_values, PX_pdf = pdf_values(x0=VLBI_data.loc[PSR_name, "px_v"], uL=VLBI_data.loc[PSR_name, "px_v_uL"], uR=VLBI_data.loc[PSR_name, "px_v_uR"])

# Calculate the joint probability distribution by multiplying the PDFs
joint_pdf = np.outer(PX_pdf, RAJ_pdf)

# Normalize the joint PDF so that it sums to 1
joint_pdf /= np.sum(joint_pdf)

# Create a meshgrid for contour plotting
X, Y = np.meshgrid(RAJ_values, PX_values)

# Create figure and axes objects for main plot and marginal distributions
sns.set_theme(context="paper", style="ticks", font_scale=2.5,
              rc={"axes.axisbelow": False, "grid.linewidth": 1.4})
fig, ([ax_marginal_x, other], [ax_main, ax_marginal_y]) = plt.subplots(2, 2, figsize=(11, 8),
                                                                       gridspec_kw={'height_ratios': [1, 4],
                                                                                    'width_ratios': [4, 1],
                                                                                    'hspace': 0.00,
                                                                                    'wspace': 0.00})

ax_marginal_x.sharex(ax_main)
ax_marginal_y.sharey(ax_main)

# Plot contour plot on main axes
contour = ax_main.contourf(X, Y, joint_pdf, cmap="viridis", levels=10, zorder=0)
plt.colorbar(contour, ax=ax_main, label='PDF', location="left", pad=-0.15,
             anchor=(-2.5, 0.5))
ax_main.set_xlabel(f"$\\alpha - $ {ref_RAJ}")

if ref_RAJ.hms[0] > 0:
    ax_main.set_xlabel("$\\alpha - " + f"{ref_RAJ:latex}"[1:-1] + "[\mathrm{mas}]$")
else:
    ax_main.set_xlabel("$\\alpha + " + f"{ref_RAJ:latex}"[2:-1] + "[\mathrm{mas}]$")

ax_main.set_ylabel("$\\varpi [\mathrm{mas}]$")

ax_main.axhline(y=1.625, color='blue', linestyle='--', lw=3)
ax_main.axvline(x=trial_RAJ_ms, color='red', linestyle='--', lw=3)

ax_main.scatter(x=trial_RAJ_ms, y=1.625, marker='x', color='black', s=500, zorder=10)

#if ref_DECJ.dms[0] > 0:
#    ax_main.set_ylabel("$\delta - " + f"{ref_DECJ:latex}"[1:-1] + "[\mathrm{mas}]$")
#else:
#    ax_main.set_ylabel("$\delta + " + f"{ref_DECJ:latex}"[2:-1] + "[\mathrm{mas}]$")

# Plot marginal distribution for X on top right subplot
ax_marginal_x.plot(RAJ_values, RAJ_pdf, color='red')
ax_marginal_x.set_xlim(ax_marginal_x.get_xlim())
ax_marginal_x.set_ylim(0, np.max(RAJ_pdf) * 1.1)
ax_marginal_x.tick_params(axis='x', bottom=False, labelbottom=False)
ax_marginal_x.set_yticks([])
ax_marginal_x.axvline(x=trial_RAJ_ms, color='red', linestyle='--', lw=3)
#    ax_marginal_x.set_title('PDF of X')

# Plot marginal distribution for Y on bottom left subplot
ax_marginal_y.plot(PX_pdf, PX_values, color='blue')
ax_marginal_y.set_ylim(ax_marginal_y.get_ylim())
ax_marginal_y.set_xlim(0, np.max(PX_pdf) * 1.1)
ax_marginal_y.tick_params(axis='y', left=False, labelleft=False)
ax_marginal_y.set_xticks([])
ax_marginal_y.axhline(y=1.625, color='blue', linestyle='--', lw=3)
#    ax_marginal_y.set_title('PDF of Y')

# Remove unnecessary spines
other.spines['right'].set_visible(False)
other.spines['top'].set_visible(False)
other.set_xticks([])
other.set_yticks([])



# Adjust layout and show plot
plt.tight_layout()
plt.savefig("./results/frame_tie/" + PSR_name + "_priors.pdf", bbox_inches='tight')
plt.show()