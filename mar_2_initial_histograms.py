from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


with fits.open("A1_mosaic.fits") as file:
    header = file[0].header
    data_map = file[0].data
    print(data_map.shape)

plt.hist(data_map.flatten(), range = (3200,3800), bins = 200)
plt.xlabel("Raw Pixel Flux Value")
plt.ylabel("Counts")
plt.title("Distribution of Pixels from Background Noise, bin width = 3 units")
plt.grid()
plt.show()

plt.hist(data_map.flatten(), range = (3800,60000), bins = 1000)
plt.xlabel("Raw Pixel Flux Value")
plt.ylabel("Counts")
plt.title("Distribution of (Approximately) non-Background Pixels, bin width = 56.2 units")
plt.xlim(3700,60000)
plt.grid()
plt.show()

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
