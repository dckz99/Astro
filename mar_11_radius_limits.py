import ast
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


with fits.open("A1_mosaic.fits") as file:
    header = file[0].header
    zero_point = header["MAGZPT"]

fig, axis = plt.subplots(figsize = (7,5))

x = np.arange(0, 25)
for offsets in x:
    y = (x - offsets) * 0.6
    plt.plot(x, y, "salmon", alpha = 0.5)
for offsets in x:
    y = (x - offsets) * 0.27
    plt.plot(x, y, "skyblue", alpha = 0.5)

data = np.loadtxt("mar_10_data_grt2_radius_limit.txt", skiprows = 2,
    delimiter = "\t")
fluxs = data[:,3]
magnitudes = zero_point - 2.5*np.log10(fluxs)
magnitudes = np.sort(magnitudes)
mag_limits = np.array([magnitudes[0]])
N = np.array([1])
for m in magnitudes[1:]:
    mag_limits = np.append(mag_limits, m)
    N = np.append(N, N[-1]+1)
log_N = np.log10(N)
axis.plot(mag_limits, log_N, "k")

data = np.loadtxt("mar_10_data_no_radius_limit.txt", skiprows = 2,
    delimiter = "\t")
fluxs = data[:,3]
magnitudes = zero_point - 2.5*np.log10(fluxs)
magnitudes = np.sort(magnitudes)
mag_limits = np.array([magnitudes[0]])
N = np.array([1])
for m in magnitudes[1:]:
    mag_limits = np.append(mag_limits, m)
    N = np.append(N, N[-1]+1)
log_N = np.log10(N)
axis.plot(mag_limits, log_N, "--k")

custom_lines=[Line2D([0],[0], color="k", lw=1.5),
            Line2D([0],[0], color="k", lw=1.5, ls="--"),
            Line2D([0],[0], color="salmon", lw=2),
            Line2D([0],[0], color="skyblue", lw=2)]
custom_labels = ["No Radius Limit", "r ≤ 2 Excluded",
    "Gradient = 0.6", "Gradient = 0.27"]
axis.legend(custom_lines, custom_labels,loc = 'lower right')

axis.set_xlabel("Calibrated Galaxy Magnitude")
axis.set_ylabel("log$_{10}$[N(<m)]")
axis.set_xlim(np.min(magnitudes), np.max(magnitudes))
axis.set_ylim(np.min(log_N), np.max(log_N)+0.03)
plt.show()
