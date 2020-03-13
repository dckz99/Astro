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

plt.imshow(np.log(data_map), cmap = "nipy_spectral", \
    vmin=np.log(gb_min), vmax = 8.5)
cbar = plt.colorbar()
cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)
plt.xlabel("Image x (relative)")
plt.ylabel("Image y (relative)")
plt.show()

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

    if upper_radius > max_radius:
        raise Exception("Maximum Radius Exceed")

    for radius in np.arange(0,upper_radius+1,1):

        if (galatic_centre[1]-radius<0)or(galatic_centre[0]-radius<0)or\
            (galatic_centre[1]+radius+1>data_map.shape[0])or\
            (galatic_centre[0]+radius+1>data_map.shape[1]):
            raise Exception("Edge of Image Reached")

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

    return total_area, total_flux, total_area_mids, gradient, \
        np.arange(0,upper_radius+1,1)


def gaussian_shape(x, mean, std):
    power = -0.5*((x-mean)/std)**2
    scale = 1/(std*np.sqrt(2*np.pi))
    return scale*np.exp(power)


def search_area(galatic_centre, max_radius):

    #pickout local space
    local_space = data_map\
        [galatic_centre[1]-max_radius:galatic_centre[1]+max_radius+1, \
        galatic_centre[0]-max_radius:galatic_centre[0]+max_radius+1]
    flat_local_space = local_space.flatten()
    range = np.max(flat_local_space) - np.min(flat_local_space)

    #bin local space data
    bin_heights, fluxs = np.histogram(flat_local_space, bins = range)
    bin_heights = np.array(bin_heights)
    fluxs = np.array(fluxs[0:-1])
    max_count = np.max(bin_heights)
    max_count_index = np.where(bin_heights == max_count)[0][0]

    #aptempt to chacacterise background by finding FWHM of guassian
    offsets = []
    for direction in [-1, +1]:
        search = max_count
        offset = 0
        while search > max_count/2:
            offset += 1
            search = bin_heights[max_count_index + offset*direction]
        offsets += [offset]
    FWHM_gauss_std = (offsets[0] + offsets[1]) / 2.355
    local_peak_index=(max_count_index-offsets[0]*2,max_count_index+offsets[1]*2)
    FWHM_gauss_mean = np.average(fluxs[local_peak_index[0]:local_peak_index[1]],
        weights = bin_heights[local_peak_index[0]:local_peak_index[1]])

    #expanding circular radius
    result = expand_apature(galatic_centre, max_radius)
    std_rad=result[4][np.where(result[3]<FWHM_gauss_mean+FWHM_gauss_std)[0][0]]
    std_area=result[2][np.where(result[3]<FWHM_gauss_mean+FWHM_gauss_std)[0][0]]

    fig = plt.figure(figsize = (14,8))

    ax1 = fig.add_subplot(2,2,1)
    plt.plot(max_radius,max_radius, "+C3")
    color_scale = ax1.imshow(np.log(local_space))
    ax1.set_xlabel("X (relative)")
    ax1.set_ylabel("Y (relative)")
    axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
        bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
    cbar = fig.colorbar(color_scale, cax = axins)
    cbar.set_label("log$_{10}$ Flux", rotation=270, labelpad = 20)
    phi = np.linspace(0, np.pi*2, num = 50)
    x = np.cos(phi)*std_rad + max_radius
    y = np.sin(phi)*std_rad + max_radius
    ax1.plot(x,y, "C3")

    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xlim(np.min(result[2]), np.max(result[2]))
    ax2.plot(result[2], result[3], \
        label = "Expanding Circular Apature About Centre", zorder = 1)
    ax2.set_xlabel("Total Apature Area (pixels)")
    ax2.set_ylabel("Change in Flux / Change in Area")
    ax2.set_ylim(FWHM_gauss_mean-FWHM_gauss_std, \
        (FWHM_gauss_mean+5*FWHM_gauss_std))
    ax2.plot([std_area, std_area], ax2.set_ylim(), \
        "C3", zorder = 0, label = "First Point Within 1 Std")
    ax2.plot([result[2][0], result[2][-1]], [FWHM_gauss_mean,FWHM_gauss_mean], \
        color = "C2", zorder = 0, label = "Mean of Background")
    rect = patches.Rectangle([result[2][0], FWHM_gauss_mean-FWHM_gauss_std],\
        result[2][-1]-result[2][0],2*FWHM_gauss_std,color="C2",\
        zorder = 0, alpha = 0.2, label = "Std of Background")
    ax2.add_patch(rect)

    plot_range = 80
    ax3 = fig.add_subplot(2,2,2)
    ax3.set_xlim(np.min(local_space), np.min(local_space)+plot_range)
    plot_range=np.arange(np.min(local_space), np.min(local_space)+plot_range+1)
    FWHM_gauss = gaussian_shape(plot_range, FWHM_gauss_mean, FWHM_gauss_std)
    ax3.hist(flat_local_space, bins = plot_range, normed = True,\
        range = (np.min(flat_local_space),np.min(flat_local_space)+plot_range),
        color = "#460e61",zorder = 1,alpha = 0.7, \
        label = "Cropped Flux Distribution\nin Local Space")
    ax3.plot(plot_range, FWHM_gauss, color = "C2", \
        label = "Gaussian Fit")
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
search_area((132, 104), 50)
search_area((101, 253), 50)
search_area((224, 215), 50)
search_area((99, 215), 50)
search_area((201, 162), 50)
search_area((189, 295), 50)
search_area((450, 61), 50)
search_area((451, 153), 50)
search_area((482, 152), 50)
search_area((245, 405), 30)
