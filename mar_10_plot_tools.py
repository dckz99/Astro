import ast
from astropy.io import fits
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
    plt.imshow(np.log10(data_map), cmap = "nipy_spectral", vmax = 8.5)
    cbar = plt.colorbar()
    cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)
    plt.xlabel("Image x (relative)")
    plt.ylabel("Image y (relative)")
    plt.show()


def plot_auto_search_result(txt_name):
    """
    Plot out galaxy map with circles showing where galxies were found.
    """
    #read in header string describing how the search was preformed
    with open(txt_name, "r") as file:
        search_params = ast.literal_eval(file.readline())

    fig = plt.figure(figsize = (11,8))
    plt.suptitle(str(search_params), wrap=True)
    ax1 = fig.add_subplot(1,1,1)
    data = np.loadtxt(txt_name, skiprows = 2, delimiter = "\t")
    theta = np.linspace(0,2*np.pi)
    x = np.cos(theta)
    y = np.sin(theta)
    for line in data:
        ax1.plot(x*line[4] + line[1], y*line[4]+ line[2], zorder=3, color="C3")

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
    cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)

    custom_lines=[Line2D([0],[0],color="C3",lw=0, marker='o',fillstyle="none"),
                    Line2D([0], [0], color="k" , lw=2, ls="--")]
    custom_labels = ["Detected Galaxy", "Detection Border"]
    ax1.legend(custom_lines, custom_labels,loc = 'upper left')
    ax1.set_xlabel("Image x (relative)")
    ax1.set_ylabel("Image y (relative)")
    plt.show()


def plot_search_iteration(data_map, galatic_centre, radius, total_area_mids,
    gradient, back_mean, back_std, aperture, search_params):
    """
    Plot data showing the analysis of a single galaxy.
    """

    loc_back_radius = search_params["loc_back_radius"]
    growth_std_multiplier = search_params["growth_std_multiplier"]

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
    ax2.plot([total_area_mids[0], total_area_mids[-1]],
    [back_mean+growth_std_multiplier*back_std,\
    back_mean+growth_std_multiplier*back_std],
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

    # plot_range = 80
    # ax3 = fig.add_subplot(2,2,3)
    # ax3.set_xlim(np.min(local_space), np.min(local_space)+plot_range)
    # plot_points=np.arange(np.min(local_space), np.min(local_space)+plot_range+1)
    # FWHM_gauss = gaussian_shape(plot_points, back_mean, back_std)
    # ax3.hist(flat_local_space, bins = plot_range, normed = True,\
    # range = (np.min(flat_local_space),np.min(flat_local_space)+plot_range),
    # color = "#460e61",zorder = 1,alpha = 0.7, \
    # label = "Cropped Flux Distribution\nin Local Space")
    # ax3.plot(plot_points, FWHM_gauss, color = "C2", \
    # label = "Gaussian Fit")
    # ax3.set_xlabel("Raw Pixel Flux Value")
    # ax3.set_ylabel("Normalised Distribution")
    # ax3.grid()
    # ax3.legend()
    # ax3.set_title("Distribution of Background Pixels")
    #
    # ax4 = fig.add_subplot(2,2,4)
    # plot_range = np.max(aperture) - np.min(aperture)
    # plot_mid = np.mean(aperture)
    # ax4.hist(aperture, bins = 200, normed = True,
    # range = (plot_mid-1000,plot_mid+1000),
    # color = "C3",zorder = 1,alpha = 0.7,
    # label = "Cropped Flux Distribution\nwithin Aperture")
    # ax4.set_title("Distribution of Aperture Pixels")
    # ax4.set_xlabel("Raw Pixel Flux Value")
    # ax4.set_ylabel("Normalised Distribution")
    # ax4.grid()
    # ax4.legend()

    back_range = 60
    ax3 = fig.add_subplot(2,1,2)

    ax3.hist(flat_local_space, bins = back_range, normed = True,\
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

def plot_catalogue(catalogue_txt):

    plot_grad = 0.27

    with fits.open("A1_mosaic.fits") as file:
        header = file[0].header
        zero_point = header["MAGZPT"]

    with open(catalogue_txt, "r") as file:
        search_params = ast.literal_eval(file.readline())

    data = np.loadtxt(catalogue_txt, skiprows = 2, delimiter = "\t")
    fluxs = data[:,3]

    magnitudes = zero_point - 2.5*np.log10(fluxs)

    # mag_range = np.max(magnitudes) - np.min(magnitudes)
    # mag_limits = np.linspace(np.min(magnitudes), np.max(magnitudes),num=1000)
    #
    # N = np.array([])
    # for ml in mag_limits:
    #     # print([True for m in magnitudes if m > ml])
    #     N = np.append(N, len([True for m in magnitudes if m <= ml]))
    # log_N = np.log10(N)
    #
    # for offsets in np.linspace(np.min(magnitudes)-mag_range,np.max(magnitudes),
    #     num = 20):
    #     y = (mag_limits - offsets)*0.6
    #     plt.plot(mag_limits, y, "k", alpha = 0.5)
    #
    # plt.plot(mag_limits, log_N)
    # plt.xlabel("Calibrated Galaxy Magnitude")
    # plt.ylabel("log$_{10}$[N(<m)]")
    # plt.xlim(np.min(magnitudes), np.max(magnitudes))
    # plt.ylim(np.min(log_N), np.max(log_N))
    # plt.show()

    magnitudes = np.sort(magnitudes)
    mag_limits = np.array([magnitudes[0]])
    N = np.array([1])
    for m in magnitudes[1:]:
        mag_limits = np.append(mag_limits, m)
        N = np.append(N, N[-1]+1)
    log_N = np.log10(N)

    fig, axis = plt.subplots(figsize = (7,5))

    for offsets in np.arange(0, np.max(magnitudes)):
        y = (mag_limits - offsets) * 0.6
        plt.plot(mag_limits, y, "salmon", alpha = 0.5)
    for offsets in np.arange(0, np.max(magnitudes)):
        y = (mag_limits - offsets) * plot_grad
        plt.plot(mag_limits, y, "skyblue", alpha = 0.5)

    custom_lines=[Line2D([0],[0], color="salmon", lw=2),
                Line2D([0],[0], color="skyblue", lw=2)]
    custom_labels = ["Gradient = 0.6", "Gradient = {}".format(plot_grad)]
    axis.legend(custom_lines, custom_labels,loc = 'lower right')

    axis.plot(mag_limits, log_N, "k")
    axis.set_xlabel("Calibrated Galaxy Magnitude")
    axis.set_ylabel("log$_{10}$[N(<m)]")
    axis.set_xlim(np.min(magnitudes), np.max(magnitudes))
    axis.set_ylim(np.min(log_N), np.max(log_N)+0.01)
    plt.show()
