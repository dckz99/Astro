from astropy.io import fits
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

# enter image coordinates of top left and bottom right pixels form ds9
tlp = (1481, 1715)
brp = (2033, 1263)

with fits.open("A1_mosaic.fits") as file:
    header = file[0].header
    data_map = np.flip(file[0].data[brp[1]:tlp[1],tlp[0]:brp[0]], 0)
    gb_min = np.min(data_map)

# plt.imshow(np.log(data_map), cmap = "nipy_spectral", \
#     vmin=np.log(gb_min), vmax = 8.5)
# cbar = plt.colorbar()
# cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)
# plt.xlabel("Image x (relative)")
# plt.ylabel("Image y (relative)")
# plt.show()


def central_displacement_map(max_radius):
    central_displacement = np.zeros([max_radius*2+1, max_radius*2+1])
    for i in range(max_radius*2+1):
        for j in range(max_radius*2+1):
            central_displacement[i,j] =\
                np.sqrt((i-max_radius)**2+(j-max_radius)**2)
    return central_displacement

max_radius = 50
central_displacement_master = central_displacement_map(max_radius)


def expand_apature(galatic_centre, upper_radius):
    total_flux = np.array([], dtype = np.int32)
    total_area = np.array([], dtype = np.int32)

    for radius in np.arange(0,upper_radius+1,1):

        central_displacement = copy(central_displacement_master\
            [max_radius-radius:max_radius+radius+1, \
            max_radius-radius:max_radius+radius+1]).flatten()

        sub_space = copy(data_map\
            [galatic_centre[1]-radius:galatic_centre[1]+radius+1, \
            galatic_centre[0]-radius:galatic_centre[0]+radius+1]).flatten()

        mask = [True if r <= radius else False for r in central_displacement]

        total_flux = np.append(total_flux, np.sum(sub_space[mask]))
        total_area = np.append(total_area, len(sub_space[mask]))

    d_total_flux = total_flux[1:] - total_flux[0:-1]
    d_total_area = total_area[1:] - total_area[0:-1]

    gradient = d_total_flux/d_total_area
    total_area_mids = (total_area[1:] + total_area[0:-1])/2

    return total_area, total_flux, total_area_mids, gradient


def gaussian_shape(x, mean, std):
    power = -0.5*((x-mean)/std)**2
    scale = 1/(std*np.sqrt(2*np.pi))
    return scale*np.exp(power)

def weighted_mean_and_std(data, weights):
    mean = np.average(data, weights = weights)
    var = np.average((data-mean)**2, weights = weights)
    return mean, np.sqrt(var)


def search_area(galatic_centre, max_radius):

    #expanding circular radius
    result = expand_apature(galatic_centre, max_radius)

    #pickout local space
    local_space = data_map\
        [galatic_centre[1]-max_radius:galatic_centre[1]+max_radius+1, \
        galatic_centre[0]-max_radius:galatic_centre[0]+max_radius+1]
    flat_local_space = local_space.flatten()
    range = np.max(flat_local_space) - np.min(flat_local_space)

    #aptempt to chacacterise background by cropping data outside +/-1std, then
    #find mean and std of cropped data
    local_mean = np.mean(flat_local_space)
    local_std = np.std(flat_local_space, ddof=1)
    lower_lim = local_mean - local_std
    upper_lim = local_mean + local_std
    crop_flat_local_space = flat_local_space[(flat_local_space >= lower_lim) \
        & (flat_local_space <= upper_lim)]
    data_range = np.linspace(np.min(crop_flat_local_space), \
        np.max(crop_flat_local_space), num = 1000)
    crop_range = np.max(crop_flat_local_space) - np.min(crop_flat_local_space)
    crop_local_mean = np.mean(crop_flat_local_space)
    crop_local_std = np.std(crop_flat_local_space, ddof=1)
    gauss = gaussian_shape(data_range, crop_local_mean, crop_local_std)

    #bin local space data
    bin_heights, fluxs = np.histogram(flat_local_space, bins = range)
    bin_heights = np.array(bin_heights)
    fluxs = np.array(fluxs[0:-1])
    max_count = np.max(bin_heights)
    max_count_index = np.where(bin_heights == max_count)[0]

    #aptempt to chacacterise background by applying 1/r weighting to data points
    #based on their seperation from the peak value and then finding their mean
    #and std
    flux_weights = bin_heights / np.abs(fluxs-fluxs[max_count_index]+0.0001)
    flux_weights[max_count_index] = (flux_weights[max_count_index+1]\
        + flux_weights[max_count_index-1])/2
    guess_mean, guess_std = weighted_mean_and_std(fluxs, flux_weights)
    guess_gauss = gaussian_shape(data_range, guess_mean, guess_std)

    #aptempt to chacacterise background by finding FWHM of guassian
    FWHM_gauss_mean = fluxs[max_count_index]
    centre_height = bin_heights[max_count_index]
    FWHM_plus_height = centre_height
    plus_offset = 0
    while FWHM_plus_height > centre_height/2:
        plus_offset += 1
        FWHM_plus_height = bin_heights[max_count_index + plus_offset]
    FWHM_minus_height = centre_height
    minus_offset = 0
    while FWHM_minus_height > centre_height/2:
        minus_offset += 1
        FWHM_minus_height = bin_heights[max_count_index - minus_offset]
    FWHM = plus_offset + minus_offset
    FWHM_gauss_std = FWHM / 2.355
    FWHM_gauss = gaussian_shape(data_range, FWHM_gauss_mean, FWHM_gauss_std)

    fig = plt.figure(figsize = (14,8))

    ax1 = fig.add_subplot(2,2,1)
    color_scale = ax1.imshow(np.log(local_space))
    ax1.set_xlabel("X (relative)")
    ax1.set_ylabel("Y (relative)")
    axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
        bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
    cbar = fig.colorbar(color_scale, cax = axins)
    cbar.set_label("log$_{10}$ Flux", rotation=270, labelpad = 20)

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(result[2], result[3], \
        label = "Expanding Circular Apature About Centre", zorder = 1)
    ax2.set_xlabel("Total Apature Area (pixels)")
    ax2.set_ylabel("Change in Flux / Change in Area")
    ax2.plot([result[2][0], result[2][-1]], [crop_local_mean,crop_local_mean], \
        color = "C3", zorder = 0, label = "Mean of Background")
    rect = patches.Rectangle([result[2][0], crop_local_mean-crop_local_std],\
        result[2][-1]-result[2][0],2*crop_local_std,color="C3",\
        zorder = 0, alpha = 0.5, label = "Std of Background")
    ax2.add_patch(rect)
    #
    # heights, _, _ = axis[1,0].hist(flat_local_space, bins = range, normed=True,\
    #     color = "C0",zorder=1,label ="Orginal Flux Distribution in Local Space")
    # rect = patches.Rectangle([lower_lim, 0],local_std*2,np.max(heights),\
    #     color="C1",zorder = 0, label = "Intial Crop of ± σ to Reach Background")
    # axis[1,0].add_patch(rect)
    # axis[1,0].set_xlabel("Raw Pixel Flux Value")
    # axis[1,0].set_ylabel("Normalised Distribution")

    ax3 = fig.add_subplot(2,1,2)
    ax3.hist(crop_flat_local_space, bins = crop_range, normed=True,\
        color = "C2",zorder = 1, \
        label = "Cropped Flux Distribution\nin Local Space")
    ax3.plot(data_range, gauss, color = "C3", \
        label = "Gaussian - mean & std of Crop")
    ax3.plot(data_range, guess_gauss, color = "C3",ls= "--", \
        label = "Gaussian - Decaying Weighting")
    ax3.plot(data_range, FWHM_gauss, color = "C3",ls= ":", \
        label = "Gaussian - Measuring FWHM")
    ax3.set_xlabel("Raw Pixel Flux Value")
    ax3.set_ylabel("Normalised Distribution")

    for ax in [ax2,ax3]:
        ax.grid()
        ax.legend()
    fig.suptitle("Searching About ({}, {}) - Radius {}"\
        .format(galatic_centre[0], galatic_centre[1], max_radius))
    fig.subplots_adjust(top = 0.94, wspace = 0.3)
    plt.show()
    fig.clear()


search_area((99, 215), 50)
search_area((189, 295), 50)
search_area((450, 61), 50)
search_area((451, 153), 50)
search_area((482, 152), 50)
