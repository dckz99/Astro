from astropy.io import fits
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def read_data():

    with fits.open("A1_mosaic.fits") as hdulist:
        header = hdulist[0].header
        data = np.flip(np.array(hdulist[0].data), axis = 0)
        # print(data)

    return data


def gaussian_shape(x, mean, std):
    power = -0.5*((x-mean)/std)**2
    scale = 1/(std*np.sqrt(2*np.pi))
    return scale*np.exp(power)

#Full width half maximum
def global_stat(data, graphs = False):

    flat_global_data = data.flatten()
    global_range = np.max(flat_global_data) - np.min(flat_global_data)

    data_range = np.linspace(3000,4000, num = 10000)

    #binning global space
    bin_heights, flux = np.histogram(flat_global_data,range = (3000,4000),
        bins = 1000)
    bin_heights[421] = bin_heights[421] - 189000
    bin_heights = np.array(bin_heights)
    flux = np.array(flux[0:-1])
    max_count = np.max(bin_heights)
    max_count_index = np.where(bin_heights == max_count)[0]

    #attempt to characterise background through FWHM of distribution
    FWHM_gauss_mean = flux[max_count_index][0]
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

    if graphs:
        plt.plot(data_range,FWHM_gauss)
        plt.hist(flux, weights = bin_heights, bins =1000,alpha =.5,density=True)
        plt.xlabel("Pixel Flux")
        plt.ylabel("Distribution")
        plt.title("Global Background Analysis - mean = {:.0f}, std = {:.1f}"\
            .format(FWHM_gauss_mean, FWHM_gauss_std))
        plt.show()

    return FWHM_gauss_mean, FWHM_gauss_std


def elliptical_mask(data_map, centre, x_rad, y_rad, ran_params):

    centre = (centre[0], data_map.shape[0] - centre[1])

    cutout = copy(data_map[centre[1]-y_rad:centre[1]+y_rad+1,
        centre[0]-x_rad:centre[0]+x_rad+1])

    for i in range(y_rad*2+1):
        y = i-y_rad
        for j in range(x_rad*2+1):
            x = j-x_rad
            if (y/y_rad)**2 + (x/x_rad)**2 <= 1:
                cutout[i,j] = np.random.normal(*ran_params)

    data_map[centre[1]-y_rad:centre[1]+y_rad+1,
        centre[0]-x_rad:centre[0]+x_rad+1] = cutout

    return data_map


def rectangle_mask(data_map, tlc, brc, ran_params):

    cutout = copy(data_map[data_map.shape[0]-tlc[1]:data_map.shape[0]-brc[1],
        tlc[0]:brc[0]])

    for i in range(cutout.shape[0]):
        for j in range(cutout.shape[1]):
            cutout[i,j] = np.random.normal(*ran_params)

    data_map[data_map.shape[0]-tlc[1]:data_map.shape[0]-brc[1],
        tlc[0]:brc[0]] = cutout

    return data_map


def noise_patch(graphs = False):
    data = read_data()
    gauss_mean, gauss_std = global_stat(data, graphs = graphs)

    if False:
        max_global = np.max(data)
        tlp = (1115,1200)#(0, 4611)#(1481, 1715)
        brp = (1150,0)#(2570, 0)

        data2 = data.copy()

        for i in range(brp[1], tlp[1]):
            for j in range(tlp[0], brp[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)


        tlp2 = (1050,1400)#(0, 4611)#(1481, 1715)
        brp2 = (1200,1200)#(2570, 0)

        for i in range(brp2[1], tlp2[1]):
            for j in range(tlp2[0], brp2[0]):
                if data2[i][j] >= max_global*0.075:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp3 = (1050,1500)#(0, 4611)#(1481, 1715)
        brp3 = (1200,1400)#(2570, 0)

        for i in range(brp3[1], tlp3[1]):
            for j in range(tlp3[0], brp3[0]):
                if data2[i][j] >= max_global*0.075:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp4 = (1050,1500)#(0, 4611)#(1481, 1715)
        brp4 = (1225,1300)#(2570, 0)

        for i in range(brp4[1], tlp4[1]):
            for j in range(tlp4[0], brp4[0]):
                if data2[i][j] >= max_global*0.050:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp5 = (1100,1700)#(0, 4611)#(1481, 1715)
        brp5 = (1145,1500)#(2570, 0)

        for i in range(brp5[1], tlp5[1]):
            for j in range(tlp5[0], brp5[0]):
                if data2[i][j] >= max_global*0.1:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp6 = (1100,3800)#(0, 4611)#(1481, 1715)
        brp6 = (1148,1700)#(2570, 0)

        for i in range(brp6[1], tlp6[1]):
            for j in range(tlp6[0], brp6[0]):
                if data2[i][j] >= max_global*0.075:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp7 = (1000,4500)#(0, 4611)#(1481, 1715)
        brp7 = (1250,3800)#(2570, 0)

        for i in range(brp7[1], tlp7[1]):
            for j in range(tlp7[0], brp7[0]):
                if data2[i][j] >= max_global*0.075:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp8 = (1000,4611)#(0, 4611)#(1481, 1715)
        brp8 = (1300,4500)#(2570, 0)

        for i in range(brp8[1], tlp8[1]):
            for j in range(tlp8[0], brp8[0]):
                if data2[i][j] >= max_global*0.075:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp9 = (1764,1410)#(0, 4611)#(1481, 1715)
        brp9 = (1825,1190)#(2570, 0)

        for i in range(brp9[1], tlp9[1]):
            for j in range(tlp9[0], brp9[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp10 = (1630,2390)#(0, 4611)#(1481, 1715)
        brp10 = (1700,2250)#(2570, 0)

        for i in range(brp10[1], tlp10[1]):
            for j in range(tlp10[0], brp10[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp11 = (1550,1920)#(0, 4611)#(1481, 1715)
        brp11 = (1630,1770)#(2570, 0)

        for i in range(brp11[1], tlp11[1]):
            for j in range(tlp11[0], brp11[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp12 = (410,910)#(0, 4611)#(1481, 1715)
        brp12 = (460,807)#(2570, 0)

        for i in range(brp12[1], tlp12[1]):
            for j in range(tlp12[0], brp12[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp13 = (410,2325)#(0, 4611)#(1481, 1715)
        brp13 = (460,2275)#(2570, 0)

        for i in range(brp13[1], tlp13[1]):
            for j in range(tlp13[0], brp13[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp14 = (460,3209)#(0, 4611)#(1481, 1715)
        brp14 = (500,3159)#(2570, 0)

        for i in range(brp14[1], tlp14[1]):
            for j in range(tlp14[0], brp14[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp15 = (80,1225)#(0, 4611)#(1481, 1715)
        brp15 = (120,1170)#(2570, 0)

        for i in range(brp15[1], tlp15[1]):
            for j in range(tlp15[0], brp15[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp16 = (990,1500)#(0, 4611)#(1481, 1715)
        brp16 = (1070,1300)#(2570, 0)

        for i in range(brp16[1], tlp16[1]):
            for j in range(tlp16[0], brp16[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp17 = (1220,1500)#(0, 4611)#(1481, 1715)
        brp17 = (1270,1300)#(2570, 0)

        for i in range(brp17[1], tlp17[1]):
            for j in range(tlp17[0], brp17[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp18 = (1025,1300)#(0, 4611)#(1481, 1715)
        brp18 = (1240,1260)#(2570, 0)

        for i in range(brp18[1], tlp18[1]):
            for j in range(tlp18[0], brp18[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp19 = (1025,1540)#(0, 4611)#(1481, 1715)
        brp19 = (1240,1500)#(2570, 0)

        for i in range(brp19[1], tlp19[1]):
            for j in range(tlp19[0], brp19[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp20 = (1250,4299)#(0, 4611)#(1481, 1715)
        brp20 = (1560,4290)#(2570, 0)

        for i in range(brp20[1], tlp20[1]):
            for j in range(tlp20[0], brp20[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp21 = (1250,4187)#(0, 4611)#(1481, 1715)
        brp21 = (1550,4160)#(2570, 0)

        for i in range(brp21[1], tlp21[1]):
            for j in range(tlp21[0], brp21[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp22 = (915,4185)#(0, 4611)#(1481, 1715)
        brp22 = (1000,4180)#(2570, 0)

        for i in range(brp22[1], tlp22[1]):
            for j in range(tlp22[0], brp22[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp23 = (870,4300)#(0, 4611)#(1481, 1715)
        brp23 = (1000,4250)#(2570, 0)

        for i in range(brp23[1], tlp23[1]):
            for j in range(tlp23[0], brp23[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

        tlp24 = (1248,4487)#(0, 4611)#(1481, 1715)
        brp24 = (1280,4484)#(2570, 0)

        for i in range(brp24[1], tlp24[1]):
            for j in range(tlp24[0], brp24[0]):
                if data2[i][j] >= max_global*0.001:
                    data2[i][j] = np.random.normal(gauss_mean, gauss_std)

    i_width, j_width = data.shape

    for i in range(0,100):
        for j in range(j_width):
            data[i,j] = np.random.normal(gauss_mean, gauss_std)

    for i in range(i_width):
        for j in range(0,100):
            data[i,j] = np.random.normal(gauss_mean, gauss_std)

    for i in range(i_width-100, i_width):
        for j in range(j_width):
            data[i,j] = np.random.normal(gauss_mean, gauss_std)

    for i in range(i_width):
        for j in range(j_width-100,j_width):
            data[i,j] = np.random.normal(gauss_mean, gauss_std)



    # #left side bright stars
    data = elliptical_mask(data, (1314,4398), 30, 30, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (1366,4330), 30, 30, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (561,4097), 30, 30, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (773,3321), 65, 120, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (1414,2980), 30, 30, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (973,2775), 50, 75, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (904,2284), 50, 75, (gauss_mean, gauss_std))

    #misc
    data = elliptical_mask(data, (1037,435), 20, 20, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (1644,342), 10, 10, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (1647,349), 5, 5, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (1533,125), 20, 20, (gauss_mean, gauss_std))

    #right side bright stars
    data = elliptical_mask(data, (1455,4032), 40, 40, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (2135,3761), 40, 60, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (2279,3946), 30, 30, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (2466,3416), 40, 40, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (2131,2309), 40, 40, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (2080,1422), 40, 60, (gauss_mean, gauss_std))
    data = elliptical_mask(data, (1773,577), 40, 40, (gauss_mean, gauss_std))

    #stem removal
    data = rectangle_mask(data, (1426,3000),(1451,0), (gauss_mean, gauss_std))
    data = rectangle_mask(data, (1423,4516),(1450,3435),(gauss_mean, gauss_std))

    #triangles
    data = rectangle_mask(data, (1388,270),(1477,213),(gauss_mean, gauss_std))
    data = rectangle_mask(data, (1388,165),(1477,115),(gauss_mean, gauss_std))
    data = rectangle_mask(data, (1321,368),(1518,237),(gauss_mean, gauss_std))
    data = rectangle_mask(data, (1321,477),(1518,425),(gauss_mean, gauss_std))
    data = rectangle_mask(data, (1687,146),(1526,121),(gauss_mean, gauss_std))
    data = rectangle_mask(data, (1099,444),(1655,424),(gauss_mean, gauss_std))
    data = rectangle_mask(data, (1013,331),(1705,310),(gauss_mean, gauss_std))
    data = rectangle_mask(data, (1287,135),(1524,121),(gauss_mean, gauss_std))


    data = elliptical_mask(data, (1425,3214), 240, 240, (gauss_mean, gauss_std))
    # data = elliptical_mask(data, (1425,3214), 240, 240, (3491, gauss_std))

    if graphs:
        fig = plt.figure()
        ax1 = fig.add_subplot()
        color_scale = ax1.imshow(np.log10(data), cmap ='nipy_spectral',vmax=3.7)

        ax1.set_xlabel("X (relative)")
        ax1.set_ylabel("Y (relative)")
        axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
            bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
        cbar = fig.colorbar(color_scale, cax = axins)
        cbar.set_label("log$_{10}$ Flux", rotation=270, labelpad = 20)
        plt.show()
    '''
    found patches of sky and filled them in with noise. All patched data should
    be stored in data2
    '''

    return data

if __name__ == "__main__":
    noise_patch()
