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

def expand_apature(galatic_centre, upper_radius):
    total_flux = []
    total_area = []
    for radius in np.arange(0,upper_radius+1,1):
        total_flux += [0]
        total_area += [0]
        central_displacement = np.zeros([radius*2+1, radius*2+1])
        sub_space = copy(data_map[galatic_centre[1]-radius:galatic_centre[1]+radius+1, \
            galatic_centre[0]-radius:galatic_centre[0]+radius+1])
        # radius = float(radius)
        for i in range(len(sub_space)):
            for j in range(len(sub_space)):
                central_displacement[i,j] = np.sqrt((i-radius)**2 + (j-radius)**2)
                if central_displacement[i,j] > radius:
                    # print(central_displacement[i,j])
                    sub_space[i,j] = 0
                else:
                    total_flux[-1] += sub_space[i,j]
                    total_area[-1] += 1

    total_flux = np.array(total_flux)
    total_area = np.array(total_area)

    d_total_flux = total_flux[1:] - total_flux[0:-1]
    d_total_area = total_area[1:] - total_area[0:-1]

    gradient = d_total_flux/d_total_area
    total_area_mids = (total_area[1:] + total_area[0:-1])/2

    return total_area, total_flux, total_area_mids, gradient

def gaussian_shape(x, mean, std):
    power = -0.5*((x-mean)/std)**2
    scale = 1/(std*np.sqrt(2*np.pi))
    return scale*np.exp(power)


def search_area(galatic_centre, radius):
    fig, axis = plt.subplots(2,2, figsize = (12,8))

    result = expand_apature(galatic_centre, radius)

    axis[0,1].plot(result[2], result[3], \
        label = "Expanding Circular Apature About Centre", zorder = 1)
    axis[0,1].grid()
    axis[0,1].set_xlabel("Total Apature Area (pixels)")
    axis[0,1].set_ylabel("Change in Flux / Change in Area")

    local_space = data_map[galatic_centre[1]-radius:galatic_centre[1]+radius+1, \
        galatic_centre[0]-radius:galatic_centre[0]+radius+1]

    color_scale = axis[0,0].imshow(np.log(local_space))
    axis[0,0].set_xlabel("X (relative)")
    axis[0,0].set_ylabel("Y (relative)")

    axins = inset_axes(axis[0,0], width="5%", height="100%", borderpad=0,\
        bbox_to_anchor=(0.1,0.,1,1), bbox_transform=axis[0,0].transAxes)

    cbar = fig.colorbar(color_scale, cax = axins)
    cbar.set_label("log$_{10}$ Flux", rotation=270, labelpad = 20)

    flat_local_space = local_space.flatten()

    range = np.max(flat_local_space) - np.min(flat_local_space)
    heights, _, _ = axis[1,0].hist(flat_local_space, bins = range, normed=True,\
        color = "C0",zorder = 1, label = "Orginal Flux Distribution in Local Space")

    local_mean = np.mean(flat_local_space)
    local_std = np.std(flat_local_space, ddof=1)

    lower_lim = local_mean - local_std
    upper_lim = local_mean + local_std

    rect = patches.Rectangle([lower_lim, 0],local_std*2,np.max(heights),color="C1",\
        zorder = 0, label = "Intial Crop of ± σ to Reach Background")
    axis[1,0].add_patch(rect)
    axis[1,0].legend()
    axis[1,0].set_xlabel("Raw Pixel Flux Value")
    axis[1,0].set_ylabel("Normalised Distribution")
    axis[1,0].grid()

    crop_flat_local_space = flat_local_space[(flat_local_space >= lower_lim) \
        & (flat_local_space <= upper_lim)]

    range = np.max(crop_flat_local_space) - np.min(crop_flat_local_space)

    axis[1,1].hist(crop_flat_local_space, bins = range, normed=True,\
        color = "C2",zorder = 1, label = "Cropped Flux Distribution in Local Space")

    local_mean = np.mean(crop_flat_local_space)
    local_std = np.std(crop_flat_local_space, ddof=1)

    data_range = np.linspace(np.min(crop_flat_local_space), \
        np.max(crop_flat_local_space), num = 100)
    gauss = gaussian_shape(data_range, local_mean, local_std)
    axis[1,1].plot(data_range, gauss, color = "C3", label = "Gaussian Form")
    axis[1,1].set_xlabel("Raw Pixel Flux Value")
    axis[1,1].set_ylabel("Normalised Distribution")
    axis[1,1].grid()
    axis[1,1].legend()

    axis[0,1].plot([result[2][0], result[2][-1]], [local_mean,local_mean], \
        color = "C3", zorder = 0, label = "Mean of Background")

    rect = patches.Rectangle([result[2][0], local_mean-local_std],\
        result[2][-1]-result[2][0],2*local_std,color="C3",\
        zorder = 0, alpha = 0.5, label = "Std of Background")
    axis[0,1].add_patch(rect)
    axis[0,1].legend()

    fig.suptitle("Searching About ({}, {}) - Radius {}"\
        .format(galatic_centre[0], galatic_centre[1], radius))
    fig.subplots_adjust(top = 0.94, wspace = 0.3)
    plt.show()


# search_area((99, 215), 50)
# search_area((189, 295), 50)
# search_area((450, 61), 50)
# search_area((451, 153), 50)
search_area((482, 152), 50)
