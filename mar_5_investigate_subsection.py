from astropy.io import fits
from copy import copy
import matplotlib.pyplot as plt
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



# galatic_centre = (99, 215)
# galatic_centre = (293, 235)
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


# galatic_centre = (99, 215)
# galatic_centre =
real_result = expand_apature((99, 215), 25)
noise_result = expand_apature((293, 235), 25)

fig, axis = plt.subplots(1,2)
axis[0].plot(real_result[0], real_result[1], label = "Known Galaxy")
axis[0].plot(noise_result[0], noise_result[1], label = "Noise Point")
axis[0].grid()
axis[0].set_xlabel("Total Apature Area (pixels)")
axis[0].set_ylabel("Total Flux")
axis[0].legend()

axis[1].plot(real_result[2], real_result[3], label = "Known Galaxy")
axis[1].plot(noise_result[2], noise_result[3], label = "Noise Point")
axis[1].grid()
axis[1].set_xlabel("Total Apature Area (pixels)")
axis[1].set_ylabel("Change in Flux / Change in Area")
axis[1].legend()
plt.show()




    # print(central_displacement)
    # print(radius)
    # print()
    # plt.imshow(central_displacement)
    # cbar = plt.colorbar()
    # cbar.set_label("Distance to Centre", rotation=270, labelpad = 20)
    # plt.xlabel("Image x (relative)")
    # plt.ylabel("Image y (relative)")
    # plt.show()
    #
    # plt.imshow(sub_space, cmap = "nipy_spectral", \
    #     vmin=0)
    # cbar = plt.colorbar()
    # cbar.set_label("log$_{10}$ Flux (Capped)", rotation=270, labelpad = 20)
    # plt.xlabel("Image x (relative)")
    # plt.ylabel("Image y (relative)")
    # plt.show()
