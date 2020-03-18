# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

hdulist = fits.open("D:\Documents\Imperial College London\Physics\Laboratory 3\Astronomical Image Processing\A1_mosaic\A1_mosaic.fits")

with fits.open("D:\Documents\Imperial College London\Physics\Laboratory 3\Astronomical Image Processing\A1_mosaic\A1_mosaic.fits") as hdulist:
    header = hdulist[0].header
    data = np.flip(np.array(hdulist[0].data))
    print(data)

# fig, axis = plt.subplots(2,2)
# axis_flat = np.ravel(axis)

#for j, i in enumerate(range(0, 60000, 1000)):    
#    axis_flat[j].hist(data.flatten(), bins = 200, range=(i,i+1000))
#    axis_flat[j].set_xlabel("Pixel Flux")
#    axis_flat[j].set_ylabel("Count")
#    axis_flat[j].grid(True)
#    
#plt.hist(data.flatten(),bins=500, range=(3000,4000))

# axis_flat[0].hist(data.flatten(),bins = 300, range=(3200,3800))
# axis_flat[0].set_xlabel("Pixel Flux")
# axis_flat[0].set_ylabel("Count")
# axis_flat[0].set_title("Histogram Plot of Pixel Flux Values from 3200 to 3800")
# axis_flat[0].grid(True)
# axis_flat[1].hist(data.flatten(),bins = 1000, range=(3800,60000))
# axis_flat[1].set_xlabel("Pixel Flux")
# axis_flat[1].set_ylabel("Count")
# axis_flat[1].set_title("Histogram Plot of Pixel Flux Values from 3800 to 60000")
# axis_flat[1].grid(True)
plt.imshow(np.log(data), cmap = 'nipy_spectral', vmax = 8.5)#[1263:1715,1481:2033]

cbar = plt.colorbar()
cbar.set_label('log$_{10}$ Flux', rotation = 270, labelpad=20)
plt.xlabel('Pixel X - axis')
plt.ylabel('Pixel Y - axis')
plt.show()
######################################################################
# plt.hist(data.flatten(), range = (3200,3800), bins = 200)
# plt.xlabel("Raw Pixel Flux Value")
# plt.ylabel("Counts")
# plt.title("Distribution of Pixels from Background Noise, bin width = 3 units")
# plt.grid()
# plt.show()

# plt.hist(data.flatten(), range = (3800,60000), bins = 1000)
# plt.xlabel("Raw Pixel Flux Value")
# plt.ylabel("Counts")
# plt.title("Distribution of (Approximately) non-Background Pixels, bin width = 56.2 units")
# plt.xlim(3700,60000)
# plt.grid()
# plt.show()
###############################################################################
# fig , axes = plt.subplots(4,5, figsize = (12,8))
#
# axes_flat = np.ravel(axes)
#
#
# # for i,range_start in enumerate(np.arange(0,20000,1000)):
# for i,range_start in enumerate(np.arange(20000,40000,1000)):
# # for i,range_start in enumerate(np.arange(40000,60000,1000)):
# # for i,range_start in enumerate(np.arange(0,20000,1000)):
#     axes_flat[i].hist(data_map.flatten(), range = (range_start,range_start+1000), bins = 200)
# # plt.xlabel("Pixel Flux Value")
# # plt.ylabel("Counts")
# # plt.title("Distribution of Pixels from Background Noise")
# plt.tight_layout()
# # fig.text(0.5, 0.96, 'Histograms of Raw Data Across Various Ranges', ha='center')
# fig.text(0.5, 0.04, 'Pixel Raw Flux Value', ha='center')
# fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical')
# fig.suptitle('Histograms of Raw Data Across Various Ranges')
# plt.show()

def gaussian_shape(x, mean, std):
    power = -0.5*((x-mean)/std)**2
    scale = 1/(std*np.sqrt(2*np.pi))
    return scale*np.exp(power)

#Full width half maximum
def global_stat():
    flat_global_data = data.flatten()
    global_mean = np.mean(flat_global_data)
    global_std = np.std(flat_global_data, ddof = 1)
    global_upperlim = global_mean + global_std
    global_lowerlim = global_mean - global_std
    crop_global_data = flat_global_data[(flat_global_data >= global_lowerlim) \
        & (flat_global_data <= global_upperlim)]
    crop_global_mean = np.mean(crop_global_data)
    crop_global_std = np.std(crop_global_data, ddof = 1)
    crop_global_range = np.max(crop_global_data) - np.min(crop_global_data)
    data_range = np.linspace(np.min(crop_global_data), \
        np.max(crop_global_data), num = 10000)
    gauss = gaussian_shape(data_range, crop_global_mean, crop_global_std)
    
    #binning global space
    bin_heights, flux = np.histogram(crop_global_data, bins = crop_global_range
                                     )
    bin_heights = np.array(bin_heights)
    flux = np.array(flux[0:-1])
    max_count = np.max(bin_heights)
    max_count_index = np.where(bin_heights == max_count)[0]
    
    #attempt to characterise background through FWHM of distribution
    FWHM_gauss_mean = flux[max_count_index]
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
    
#    fig = plt.figure(figsize = (14, 8))
#
#    ax1 = fig.add_subplot(2,2,1)
#    color_scale = ax1.imshow(np.log(data))
#    ax1.set_xlabel("X (relative)")
#    ax1.set_ylabel("Y (relative)")
#    axins = inset_axes(ax1, width="5%", height="100%", borderpad=0,\
#        bbox_to_anchor=(0.1,0.,1,1), bbox_transform=ax1.transAxes)
#    cbar = fig.colorbar(color_scale, cax = axins)
#    cbar.set_label("log$_{10}$ Flux", rotation=270, labelpad = 20)
#
#    ax3 = fig.add_subplot(2,1,2)
#    ax3.hist(crop_global_data, bins = crop_global_range, normed=True,\
#        color = "C2",zorder = 1, \
#        label = "Cropped Flux Distribution\nin Local Space")
#    # ax3.plot(data_range, gauss, color = "C3", \
#    #     label = "Gaussian - mean & std of Crop")
#    ax3.plot(data_range, FWHM_gauss, color = "C3",ls= ":", \
#        label = "Gaussian - Measuring FWHM")
#    ax3.set_xlabel("Raw Pixel Flux Value")
#    ax3.set_ylabel("Normalised Distribution")
#
#    for ax in [ax3]:
#        ax.grid()
#        ax.legend()
#    # fig.suptitle("Searching About ({}, {}) - Radius {}"\
#    #     .format(galatic_centre[0], galatic_centre[1], max_radius))
#    fig.subplots_adjust(top = 0.94, wspace = 0.3)
#    plt.show()
#    fig.clear()
    return FWHM_gauss_mean, FWHM_gauss_std

def noise_patch():
    gauss_stat = global_stat()
    gauss_mean = gauss_stat[0]
    gauss_std = gauss_stat[1]
    
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
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot()
    #color_scale = ax1.imshow(np.log10(data2))#[brp12[1]:tlp12[1],tlp12[0]:brp12[0]]
    color_scale = ax1.imshow(np.log(data2), cmap = 'nipy_spectral', vmax = 8.5)
    
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
noise_patch()