from astropy.io import fits
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import numpy as np
import numpy.random as npran


def central_displacement_map(max_radius):
    central_displacement = np.zeros([max_radius*2+1, max_radius*2+1])
    for i in range(max_radius*2+1):
        for j in range(max_radius*2+1):
            central_displacement[i,j] =\
                np.sqrt((i-max_radius)**2+(j-max_radius)**2)
    return central_displacement


def gaussian_shape(x, mean, std):
    power = -0.5*((x-mean)/std)**2
    scale = 1/(std*np.sqrt(2*np.pi))
    return scale*np.exp(power)


def local_background(galatic_centre, max_radius):
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

    return FWHM_gauss_mean, FWHM_gauss_std


def expand_apature_to_threshold(galatic_centre, back_mean, back_std,
    std_multiplier):
    #expand some search radius until growth is below threshold
    total_flux = np.array([], dtype = np.int32)
    total_area = np.array([], dtype = np.int32)
    radius = 0
    while True:
        radius += 1
        if radius > max_radius:
            raise Exception("Maximum Radius Exceed")

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

        if radius > 1:
            d_total_flux = total_flux[1:] - total_flux[0:-1]
            d_total_area = total_area[1:] - total_area[0:-1]
            gradient = d_total_flux/d_total_area

            if gradient[-1] < back_mean + back_std * std_multiplier:
                total_area_mids = (total_area[1:] + total_area[0:-1])/2
                break
    return radius, total_flux, total_area_mids,\
        np.array(mask).reshape(2*radius+1,2*radius+1), gradient, \
        sub_space[mask], total_area

# enter image coordinates of top left and bottom right pixels form ds9
tlp = (1481, 1715)
brp = (2033, 1263)

border = 30
loc_back_radius = 30
max_radius = 30

selection_std_multiplier = 5
growth_std_multiplier = 2

head_str = "name\tg_x\tg_y\tt_flux\t\tradius\tb_mean\t\tb_std"
txt_str = head_str + "\n"
print(head_str)

with fits.open("A1_mosaic.fits") as file:
    header = file[0].header
    data_map = np.flip(file[0].data[brp[1]:tlp[1],tlp[0]:brp[0]], 0)
    interior_box = copy(data_map[border:-border, border:-border])
    gb_min = np.min(data_map)

central_displacement_master = central_displacement_map(max_radius)

# plt.imshow(np.log(data_map), cmap = "nipy_spectral", \
#     vmin=np.log(gb_min), vmax = 8.5)
# cbar = plt.colorbar()
# cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)
# plt.xlabel("Image x (relative)")
# plt.ylabel("Image y (relative)")
# plt.show()

num_found = 0
while True:
    #get center point of galaxy
    max_flux = np.max(interior_box)
    max_point = np.where(interior_box == max_flux)
    max_point_x = max_point[1][0]+border
    max_point_y = max_point[0][0]+border
    galatic_centre = (max_point_x, max_point_y)

    #characterise background
    back_mean, back_std = local_background(galatic_centre,
        loc_back_radius)

    if max_flux < back_mean + selection_std_multiplier * back_std:
        break

    try:
        #expand circular apature to threshold
        radius, total_flux, total_area_mids, mask, gradient,aperture,total_area\
            = expand_apature_to_threshold(\
            galatic_centre, back_mean, back_std, growth_std_multiplier)

    except:
        print("Failed at ({},{})".format(*galatic_centre))
        raise

    # fig = plt.figure(figsize = (14,8))
    #
    # local_space = data_map\
    #     [galatic_centre[1]-loc_back_radius:galatic_centre[1]+loc_back_radius+1,\
    #     galatic_centre[0]-loc_back_radius:galatic_centre[0]+loc_back_radius+1]
    # flat_local_space = local_space.flatten()
    #
    # ax1 = fig.add_subplot(2,2,1)
    # plt.plot([loc_back_radius], [loc_back_radius],"+", markersize=1000,
    #          markeredgewidth=0.5, markeredgecolor="w")
    # # color_scale = ax1.imshow(np.log(local_space), vmin = np.log(gb_min),
    # #     vmax = np.log(max_flux), cmap = "nipy_spectral")
    # color_scale = ax1.imshow(local_space, cmap = "nipy_spectral")
    # ax1.set_xlabel("X (relative)")
    # ax1.set_ylabel("Y (relative)")
    # axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
    #     bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
    # cbar = fig.colorbar(color_scale, cax = axins)
    # # cbar.set_label("log$_{10}$ Flux", rotation=270, labelpad = 20)
    # cbar.set_label("Flux", rotation=270, labelpad = 20)
    # phi = np.linspace(0, np.pi*2, num = 50)
    # x = np.cos(phi)*radius + loc_back_radius
    # y = np.sin(phi)*radius + loc_back_radius
    # ax1.plot(x,y, "C3")
    # ax1.set_title("Object at ({}, {}), radius = {}"\
    #     .format(*galatic_centre, radius))
    #
    # ax2 = fig.add_subplot(2,2,2)
    # ax2.set_xlim(np.min(total_area_mids), np.max(total_area_mids))
    # ax2.plot(total_area_mids, gradient, zorder = 1)
    # ax2.set_xlabel("Total Apature Area (pixels)")
    # ax2.set_ylabel("Change in Flux / Change in Area")
    # ax2.set_ylim(back_mean-back_std, None)
    # ax2.plot([total_area_mids[0], total_area_mids[-1]],
    #     [back_mean+growth_std_multiplier*back_std,\
    #     back_mean+growth_std_multiplier*back_std],
    #     color = "C3", zorder = 0, label = "Growth Cutoff")
    # ax2.plot([total_area_mids[0], total_area_mids[-1]], [back_mean,back_mean],
    #     color = "C2", zorder = 0, label = "Mean of Background")
    # rect = patches.Rectangle([total_area_mids[0], back_mean-back_std],\
    #     total_area_mids[-1]-total_area_mids[0],2*back_std,color="C2",\
    #     zorder = 0, alpha = 0.2, label = "Std of Background")
    # ax2.add_patch(rect)
    # ax2.grid()
    # ax2.legend()
    # ax2.set_title("Expanding Circular Aperture")
    #
    # plot_range = 80
    # ax3 = fig.add_subplot(2,2,3)
    # ax3.set_xlim(np.min(local_space), np.min(local_space)+plot_range)
    # plot_points=np.arange(np.min(local_space), np.min(local_space)+plot_range+1)
    # FWHM_gauss = gaussian_shape(plot_points, back_mean, back_std)
    # ax3.hist(flat_local_space, bins = plot_range, normed = True,\
    #     range = (np.min(flat_local_space),np.min(flat_local_space)+plot_range),
    #     color = "#460e61",zorder = 1,alpha = 0.7, \
    #     label = "Cropped Flux Distribution\nin Local Space")
    # ax3.plot(plot_points, FWHM_gauss, color = "C2", \
    #     label = "Gaussian Fit")
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
    #     range = (plot_mid-1000,plot_mid+1000),
    #     color = "C3",zorder = 1,alpha = 0.7,
    #     label = "Cropped Flux Distribution\nwithin Aperture")
    # ax4.set_title("Distribution of Aperture Pixels")
    # ax4.set_xlabel("Raw Pixel Flux Value")
    # ax4.set_ylabel("Normalised Distribution")
    # ax4.grid()
    # ax4.legend()
    #
    # plt.subplots_adjust(hspace = 0.33)
    # plt.show()

    for i in range(len(mask)):
        clear_y = max_point_y - border - radius + i
        for j in range(len(mask)):
            clear_x = max_point_x - border - radius + j

            if mask[i,j]:
                data_map[clear_y + border, clear_x + border] = \
                    npran.normal(loc=back_mean, scale=back_std)

                if not ((clear_x < 0)or(clear_y < 0)or\
                    (clear_x >= interior_box.shape[1])or\
                    (clear_y >= interior_box.shape[0])):

                    interior_box[clear_y, clear_x] = 1

    galaxy_flux = total_flux[-1] - back_mean*total_area[-1]
    data_str = "{}\t{}\t{}\t{:.7e}\t{}\t{:.7e}\t{:.7e}"\
        .format(num_found,*galatic_centre, galaxy_flux, radius,
        back_mean, back_std)
    txt_str = txt_str + data_str + "\n"
    num_found += 1
    print(data_str)
    if num_found == 40:
        break

with open("mar_9_data.txt", "w+") as file:
    file.write(txt_str[:-1])

print("\n{} Galaxies Found".format(num_found))

# plt.imshow(np.log(data_map), cmap = "nipy_spectral", \
#     vmin=np.log(gb_min), vmax = 8.5)
# cbar = plt.colorbar()
# cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)
# plt.xlabel("Image x (relative)")
# plt.ylabel("Image y (relative)")
# plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig = plt.figure(figsize = (11,8))
ax1 = fig.add_subplot(1,1,1)

data = np.loadtxt("mar_9_data.txt", skiprows = 1, delimiter = "\t")
theta = np.linspace(0,2*np.pi)
x = np.cos(theta)
y = np.sin(theta)
for line in data:
    ax1.plot(x*line[4] + line[1], y*line[4]+ line[2], zorder = 3, color = "C3")

plt.plot(
    [border,data_map.shape[1]-border,data_map.shape[1]-border,border,border],
    [border,border, data_map.shape[0]-border,data_map.shape[0]-border,border],
    "--k", zorder = 2)

with fits.open("A1_mosaic.fits") as file:
    header = file[0].header
    data_map = np.flip(file[0].data[brp[1]:tlp[1],tlp[0]:brp[0]], 0)
    gb_min = np.min(data_map)
color_scale = ax1.imshow(np.log(data_map), cmap = "nipy_spectral", \
    vmin=np.log(gb_min), vmax = 8.5, zorder = 1)
axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
    bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
cbar = fig.colorbar(color_scale, cax = axins)
cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)

custom_lines = [Line2D([0], [0], color="C3", lw=0, marker='o',fillstyle="none"),
                Line2D([0], [0], color="k" , lw=2, ls="--")]
custom_labels = ["Detected Galaxy", "Detection Border"]
ax1.legend(custom_lines, custom_labels,loc = 'upper left')
ax1.set_xlabel("Image x (relative)")
ax1.set_ylabel("Image y (relative)")
plt.show()
