import ast
from astropy.io import fits
import csv
from lmfit import Model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import numpy as np


def gaussian_shape(x, mean, std):
    """
    Simple gaussian function.
    """
    power = -0.5*((x-mean)/std)**2
    scale = 1/(std*np.sqrt(2*np.pi))

    return scale*np.exp(power)


def plot_image(data_map):
    """
    Show uneditted data.
    """
    plt.imshow(np.log10(data_map), cmap = "nipy_spectral", vmax = 3.7)
    cbar = plt.colorbar()
    cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)
    plt.xlabel("Image x (relative)")
    plt.ylabel("Image y (relative)")
    plt.show()


def plot_auto_search_result(csv_name):
    """
    Plot out galaxy map with circles showing where galxies were found.
    """
    fig = plt.figure(figsize = (11,8))
    ax1 = fig.add_subplot(1,1,1)

    theta = np.linspace(0,2*np.pi)
    x = np.cos(theta)
    y = np.sin(theta)

    with open(csv_name, 'r', newline='') as file:
        reader = csv.reader(file)
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                #read in header string describing how the search was preformed
                search_params = ast.literal_eval(*row)
            elif count == 1:
                count += 1
            else:
                ax1.plot(x*float(row[5])+float(row[1]),
                    y*float(row[5])+float(row[2]), zorder=3, color="C3")

    border = search_params["border"]
    tlp = search_params["tlp"]
    brp = search_params["brp"]
    x_width = brp[0]-tlp[0]
    y_width = tlp[1]-brp[1]

    plt.plot(
        [border, x_width-border, x_width-border, border, border],
        [border, border, y_width-border, y_width-border, border],
        "--k", zorder = 2)

    with fits.open("A1_mosaic.fits") as file:
        header = file[0].header
        data_map = np.flip(file[0].data[brp[1]:tlp[1],tlp[0]:brp[0]], 0)
        gb_min = np.min(data_map)
    # color_scale = ax1.imshow(np.log10(data_map), cmap = "nipy_spectral",
    #     vmin = np.min(np.log10(data_map)), vmax = 3.58, zorder = 1)
    color_scale = ax1.imshow(data_map, cmap = "nipy_spectral", zorder = 1,
        vmax = 3600)
    # color_scale = ax1.imshow(data_map, cmap = "nipy_spectral", zorder = 1,
    #     vmax = 4000)
    axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
        bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
    cbar = fig.colorbar(color_scale, cax = axins)
    cbar.set_label("Flux (Capped)", rotation=270, labelpad = 20)

    custom_lines=[Line2D([0],[0],color="C3",lw=0, marker='o',fillstyle="none"),
                    Line2D([0], [0], color="k" , lw=2, ls="--")]
    custom_labels = ["Detected Galaxy", "Detection Border"]
    ax1.legend(custom_lines, custom_labels,loc = 'upper left')
    ax1.set_xlabel("Image x (relative)")
    ax1.set_ylabel("Image y (relative)")
    plt.suptitle(str(search_params), wrap=True)
    plt.show()


def plot_search_iteration(data_map, galatic_centre, radius, total_area,
    gradient, back_mean, back_std, aperture, search_params):
    """
    Plot data showing the analysis of a single galaxy.
    """

    loc_back_radius = search_params["loc_back_radius"]
    growth_std_multiplier = search_params["growth_std_multiplier"]

    total_area_mids = (total_area[1:] + total_area[0:-1])/2

    fig = plt.figure(figsize = (13,8.1))

    local_space = data_map\
    [galatic_centre[1]-loc_back_radius:galatic_centre[1]+loc_back_radius+1,\
    galatic_centre[0]-loc_back_radius:galatic_centre[0]+loc_back_radius+1]
    flat_local_space = local_space.flatten()

    ax1 = fig.add_subplot(2,2,1)
    plt.plot([loc_back_radius], [loc_back_radius],"+", markersize=1000,
         markeredgewidth=0.5, markeredgecolor="w")
    # color_scale = ax1.imshow(np.log10(local_space), vmin = np.log10(gb_min),
    #     vmax = np.log10(max_flux), cmap = "nipy_spectral")
    color_scale = ax1.imshow(local_space, cmap = "nipy_spectral")
    ax1.set_xlabel("X (relative)")
    ax1.set_ylabel("Y (relative)")
    axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
    bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
    cbar = fig.colorbar(color_scale, cax = axins)
    # cbar.set_label("log$_{10}$ Flux", rotation=270, labelpad = 20)
    cbar.set_label("Flux", rotation=270, labelpad = 20)
    phi = np.linspace(0, np.pi*2, num = 50)
    x = np.cos(phi)*radius + loc_back_radius
    y = np.sin(phi)*radius + loc_back_radius
    ax1.plot(x,y, "C3")
    ax1.set_title("Object at ({}, {}), radius = {}"\
    .format(*galatic_centre, radius))

    ax2 = fig.add_subplot(2,2,2)
    ax2.set_xlim(np.min(total_area_mids), np.max(total_area_mids))
    ax2.plot(total_area_mids, gradient, zorder = 1)
    ax2.set_xlabel("Total Apature Area (pixels)")
    ax2.set_ylabel("Change in Flux / Change in Area")
    ax2.set_ylim(back_mean-back_std, None)
    dA = total_area[1:] - total_area[0:-1]
    ax2.plot(total_area_mids,
    back_mean+growth_std_multiplier*back_std/np.sqrt(dA),
    color = "C3", zorder = 0, label = "Growth Cutoff")
    ax2.plot([total_area_mids[0], total_area_mids[-1]], [back_mean,back_mean],
    color = "C2", zorder = 0, label = "Mean of Background")
    rect = patches.Rectangle([total_area_mids[0], back_mean-back_std],\
    total_area_mids[-1]-total_area_mids[0],2*back_std,color="C2",\
    zorder = 0, alpha = 0.2, label = "Std of Background")
    ax2.add_patch(rect)
    ax2.grid()
    ax2.legend()
    ax2.set_title("Expanding Circular Aperture")


    back_range = 60
    ax3 = fig.add_subplot(2,1,2)
    ax3.hist(flat_local_space, bins = back_range, density = True,\
    range = (np.min(flat_local_space),np.min(flat_local_space)+back_range),
    color = "#460e61",zorder = 1, alpha = 0.7, \
    label = "Cropped Background Flux\nDistribution in Local Space")
    plot_points=np.linspace(*ax3.set_xlim(), num = 80)
    FWHM_gauss = gaussian_shape(plot_points, back_mean, back_std)
    ax3.plot(plot_points, FWHM_gauss, color = "C2", label = "Gaussian Fit")
    aperture_range = 100
    counts, bins = np.histogram(aperture, bins = aperture_range,
        range = (np.min(aperture), np.min(aperture) + aperture_range))
    ax3.hist(bins[:-1], bins, weights = counts/len(aperture), color = "C1",
        label = "Crop Flux Distribution\nin Aperture",alpha = 0.7)
    ax3.set_xlabel("Raw Pixel Flux Value")
    ax3.set_ylabel("Normalised Distribution")
    ax3.grid()
    ax3.legend()
    ax3.set_title("Pixel Distribution")
    ax3.set_xlim(np.min(flat_local_space), np.min(aperture)+aperture_range)

    plt.subplots_adjust(hspace = 0.33, wspace = 0.31)
    plt.show()


def plot_catalogue(catalogue_csv):

    plot_grad = 0.27

    with open(catalogue_csv, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            #read in header string describing how the search was preformed
            search_params = ast.literal_eval(*row)
            break

    data = np.loadtxt(catalogue_csv, skiprows = 2, delimiter = ",")

    stacked_data = np.stack((data[:,3],data[:,4]), axis=1)
    stacked_data = stacked_data[stacked_data[:, 0].argsort()]

    magnitudes = stacked_data[:,0]
    magnitudes_e = stacked_data[:,1]
    # mag_limits = np.array([magnitudes[0]])
    N = np.array([1])
    for m in magnitudes[1:]:
        # mag_limits = np.append(mag_limits, m)
        N = np.append(N, N[-1]+1)
    N_e = np.sqrt(N)
    log_N = np.log10(N)
    log_N_e = N_e /(np.log(10)*N)

    weights = 1/np.sqrt((log_N_e/log_N)**2+(magnitudes_e/magnitudes)**2)
    for i in range(len(magnitudes)):
        if magnitudes[i] > 16.3:
            weights[i] = 0


    fig, axis = plt.subplots(figsize = (7,5))


    # for offsets in np.arange(0, np.max(magnitudes)):
    #     y = (mag_limits - offsets) * 0.6
    #     plt.plot(mag_limits, y, "salmon", alpha = 0.5)
    # for offsets in np.arange(0, np.max(magnitudes)):
    #     y = (mag_limits - offsets) * plot_grad
    #     plt.plot(mag_limits, y, "skyblue", alpha = 0.5)

    # custom_lines=[Line2D([0],[0], color="salmon", lw=2),
    #             Line2D([0],[0], color="skyblue", lw=2)]
    # custom_labels = ["Gradient = 0.6", "Gradient = {}".format(plot_grad)]
    # axis.legend(custom_lines, custom_labels,loc = 'lower right')

    # axis.errorbar(mag_limits, log_N, xerr= magnitudes_e,
    #     fmt = "none", color = "salmon", alpha = 0.5)
    axis.fill_between(magnitudes, y1 = log_N+log_N_e, y2 = log_N-log_N_e,
        color = "salmon", alpha = 0.5, label = "log$_{10}$(N) Error")
    # axis.errorbar(mag_limits, log_N, yerr= log_N_e,
    #     fmt = "none", color = "skyblue", alpha = 0.5)
    axis.fill_betweenx(log_N, x1 = magnitudes-magnitudes_e,
        x2 = magnitudes+magnitudes_e, color = "skyblue", alpha = 0.7,
        label = "Magnitude Error")
    axis.plot(magnitudes, log_N, "k")

    def linear(x, gradient, intercept):
        return x*gradient + intercept
    lmodel = Model(linear)
    result = lmodel.fit(log_N, x=magnitudes, gradient=0.6, intercept=-2.5,
        weights = weights)
    print(result.fit_report())
    axis.plot(magnitudes, result.best_fit, 'k--', label='Best Fit')

    axis.set_xlabel("Calibrated Galaxy Magnitude")
    axis.set_ylabel("log$_{10}$[N(<m)]")
    axis.set_xlim(np.min(magnitudes), np.max(magnitudes))
    axis.set_ylim(np.min(log_N), np.max(log_N)+0.03)
    axis.legend()
    plt.show()
