from astropy.io import fits
from copy import copy
import csv
from lmfit import Model
import numpy as np
import numpy.random as npran

from mar_16_plot_tools import *


def central_displacement_map(max_radius):
    """
    Generates square 2D array with the value of each cell being the cell's
    seperation from the central cell. This is generated here to avoid
    recalculation.

    max_radius is the maximum radius circle that could fit in this square.
    """
    central_displacement = np.zeros([max_radius*2+1, max_radius*2+1],
        dtype = np.float64)
    for i in range(max_radius*2+1):
        for j in range(max_radius*2+1):
            central_displacement[i,j] =\
                np.sqrt((i-max_radius)**2+(j-max_radius)**2)

    return central_displacement


def local_background(data_map, galatic_centre, loc_back_radius):
    """
    Investigates the background noise level in a square
    (width = loc_back_radius*2+1) about the galatic centre.
    """
    #pickout local space which the background is investigated in
    local_space = data_map\
        [galatic_centre[1]-loc_back_radius:galatic_centre[1]+loc_back_radius+1,\
        galatic_centre[0]-loc_back_radius:galatic_centre[0]+loc_back_radius+1]
    flat_local_space = local_space.flatten()

    #bin local space data, each bar has integer width -> no information loss
    range = np.max(flat_local_space) - np.min(flat_local_space)
    bin_heights, fluxs = np.histogram(flat_local_space, bins = range)
    bin_heights = np.array(bin_heights)
    fluxs = np.array(fluxs[0:-1])
    max_count = np.max(bin_heights)
    max_count_index = np.where(bin_heights == max_count)[0][0]
    #pick out heighest bin and find the relevant index in fluxs
    #in most cases this bin will be part of the noise

    assert 3390 < fluxs[max_count_index] < 3450

    #background is chacacterised by finding the FWHM of the data
    offsets = []
    for direction in [-1, +1]: #walks in different directions
        search = max_count
        offset = 0
        while search > max_count/2: #find half maxima
            offset += 1
            search = bin_heights[max_count_index + offset*direction]
        offsets += [offset]
    #flux values have integer spacing -> index diffence = flux difference
    FWHM_gauss_std = (offsets[0] + offsets[1]) / 2.355
    local_peak_index=(max_count_index-offsets[0]*2,max_count_index+offsets[1]*2)
    FWHM_gauss_mean = np.average(fluxs[local_peak_index[0]:local_peak_index[1]],
        weights = bin_heights[local_peak_index[0]:local_peak_index[1]])
    #find mean of gaussian by averaging histogram bars in area around peak

    return FWHM_gauss_mean, FWHM_gauss_std


def expected_df_dA_func(area, scale, width):
    """
    Functional model of the expected form of df/dA as the aperture expands.
    """
    return (scale) * np.exp(-area/width)


def expand_apature_to_threshold(data_map, galatic_centre, back_mean, back_std,
    growth_std_multiplier, max_radius, central_displacement_master,
    individual_plots = False):
    """
    Expand cicular aperture about galatic_centre, until the rate of change of
    total flux passing through the aperture wrt area of the aperture falls below
    the threshold defined by growth_std_multiplier.

    Once below this threshold the pixels that the aperture is growing into are
    considered to be part of the background noie (pixel area = 1).
    """
    radius = 0

    total_flux = np.array([data_map[galatic_centre[1],galatic_centre[0]]],
        dtype = np.int32)
    total_area = np.array([1], dtype = np.int32)
    gradient   = np.array([], dtype = np.float64)

    #increase the radius
    while True:
        radius += 1
        if radius > max_radius:
            raise Exception("Maximum Radius Exceed")
        if (galatic_centre[1]-radius<0)or(galatic_centre[0]-radius<0)or\
            (galatic_centre[1]+radius+1>data_map.shape[0])or\
            (galatic_centre[0]+radius+1>data_map.shape[1]):
            raise Exception("Edge of Image Reached")

        #slice out relevant part of central_displacement_master to give array
        #of correct size where the value of every element is that element's
        #distance form the central element
        central_displacement = copy(central_displacement_master\
            [max_radius-radius:max_radius+radius+1, \
            max_radius-radius:max_radius+radius+1]).flatten()

        #slice out space that aperture occupies
        sub_space = copy(data_map\
            [galatic_centre[1]-radius:galatic_centre[1]+radius+1, \
            galatic_centre[0]-radius:galatic_centre[0]+radius+1]).flatten()

        #use mask of T/F values to slice elements from the array that are within
        #aperture
        mask = [True if r <= radius else False for r in central_displacement]
        total_flux = np.append(total_flux, np.sum(sub_space[mask]))
        total_area = np.append(total_area, len(sub_space[mask]))

        if True:
            #calculate rate of change of flux
            d_total_flux = total_flux[-1] - total_flux[-2]
            d_total_area = total_area[-1] - total_area[-2] #single pixel area=1
            gradient = np.append(gradient, d_total_flux / d_total_area)

            #if growth rate is increasing
            if radius > 1 and gradient[-1] > gradient[-2]:
                #levave with results from previous radius
                total_area = total_area[:-1]
                break
            else:
                #update and store lastest result
                return_line = (radius, copy(total_flux),\
                  np.array(mask).reshape(2*radius+1,2*radius+1),copy(gradient),\
                  sub_space[mask], copy(total_area))

            assert d_total_area > 0
            #growth is below threshold
            if gradient[-1] < back_mean + \
                (back_std * growth_std_multiplier)/np.sqrt(d_total_area):
                break

    radius,total_flux,mask,gradient,aperture,total_area = return_line
    total_area_mids = (total_area[1:] + total_area[0:-1])/2

    if radius > 2:
        #fit exponential to df_dA to estimate flux cutoff by finite aperture
        total_area_mids = (total_area[1:] + total_area[0:-1])/2
        gmodel = Model(expected_df_dA_func)
        result = gmodel.fit(gradient - back_mean, area=total_area_mids,
            scale=5000, width=10)
        width = result.best_values["width"]
        width_e = np.sqrt(result.covar[1,1])
        scale = result.best_values["scale"]
        scale_e = np.sqrt(result.covar[0,0])

        cutoff_correction = (scale*width)*np.exp(-total_area[-1]/width)
        cutoff_correction_e = np.sqrt((width*scale_e)**2\
            +(scale*width_e*(1+total_area[-1]/width))**2)\
            *np.exp(-total_area[-1]/width)


        final_flux = total_flux[-1]
        final_area = total_area[-1]
        #total flux, minus background noise
        galaxy_flux = final_flux - back_mean*final_area

        back_removal_e = back_std*np.sqrt(final_area)
        expected_count_e = np.sqrt(galaxy_flux)

        error_data = (cutoff_correction, cutoff_correction_e, back_removal_e,
            expected_count_e)

    else:
        final_flux = total_flux[-1]
        final_area = total_area[-1]
        return radius, final_flux, final_area, mask, None

    if individual_plots and radius > 2:
        #plot single galaxy
        plot_search_iteration(data_map, galatic_centre,radius,
            total_area,gradient,back_mean,back_std,aperture,
            search_params)

    return radius, final_flux, final_area, mask, error_data


def auto_search(csv_name, tlp, brp, border, loc_back_radius, max_radius,
    selection_std_multiplier, growth_std_multiplier, limit = None,
    initial_plot = False, final_plot = False, individual_plots = False):
    """
    Find new galaxies by selecting the brightest pixel in the image and then
    expanding a circular aperture to encompass the whole galaxy.

    Output result to terminal and txt file.
    """

    assert border >= loc_back_radius
    assert border >= max_radius

    search_params = {
        "tlp" : tlp,
        "brp" : brp,
        "border" : border,
        "loc_back_radius" : loc_back_radius,
        "max_radius" : max_radius,
        "selection_std_multiplier" : selection_std_multiplier,
        "growth_std_multiplier" : growth_std_multiplier}

    #first line of txt describe the galaxy search next line is column headings
    head_str = "name\tg_x\tg_y\tradius\tmag\terror"
    print(head_str)

    with fits.open("A1_mosaic.fits") as file: #read in data
        header = file[0].header
        zero_point = header["MAGZPT"]
        #full data map
        data_map = np.flip(file[0].data[brp[1]:tlp[1],tlp[0]:brp[0]], 0)
        #slice out box from data_map, excluding borders
        interior_box = copy(data_map[border:-border, border:-border])
        gb_min = np.min(data_map)

    #create displacement map
    central_displacement_master = central_displacement_map(max_radius)

    if initial_plot: #plot data, pre analysis
        plot_image(data_map)

    num_found = 0
    all_data = []
    data_str = ""
    while True:
        #get centre point of new galaxy by finding brighest pixl in interior box
        max_flux = np.max(interior_box)
        max_point = np.where(interior_box == max_flux)
        max_point_x = max_point[1][0]+border #coord shift back to data_map
        max_point_y = max_point[0][0]+border
        galatic_centre = (max_point_x, max_point_y)

        try:
            #characterise background
            back_mean, back_std = local_background(data_map, galatic_centre,
                loc_back_radius)

            #test that centre point is above background noise
            if max_flux < back_mean + selection_std_multiplier * back_std:
                interior_box[max_point_y-border, max_point_x-border] = 1
                # print(" \t{}\t{}\t".format(*galatic_centre))
                if max_flux < back_mean + back_std*selection_std_multiplier/2:
                    try:
                        print(data_str[:-1])
                    finally:
                        break #stop searching
                continue

            #expand circular apature upto threshold
            radius, final_flux, final_area, mask, error_data\
                = expand_apature_to_threshold(data_map,galatic_centre,back_mean,
                back_std, growth_std_multiplier, max_radius,
                central_displacement_master, individual_plots=individual_plots)

        except:
            print("\nFailed at ({},{})\n".format(*galatic_centre))
            raise

        #clear out pixels that have already been considered
        for i in range(len(mask)):
            clear_y = galatic_centre[1] - border - radius + i
            for j in range(len(mask)):
                clear_x = galatic_centre[0] - border - radius + j

                if mask[i,j]:
                    #set pixels within cicular aperture to random noise so they
                    #are not reconsidered but don't skew noise chacacterisation
                    data_map[clear_y + border, clear_x + border] = \
                        npran.normal(loc=back_mean, scale=back_std)

                    #interior_box is only used to find new galaxies, set pixels
                    #to 1
                    if not ((clear_x < 0)or(clear_y < 0)or\
                        (clear_x >= interior_box.shape[1])or\
                        (clear_y >= interior_box.shape[0])):
                        interior_box[clear_y, clear_x] = 1

        galaxy_flux = final_flux - back_mean*final_area
        if radius > 2 and galaxy_flux > 0:
            cutoff_correction, cutoff_correction_e, back_removal_e, \
                expected_count_e = error_data

            corrected_flux = galaxy_flux + cutoff_correction

            corrected_flux_e = np.sqrt(expected_count_e**2 \
                + cutoff_correction_e**2 + back_removal_e**2)

            cal_mag = zero_point - 2.5*np.log10(corrected_flux)
            cal_mag_e = (2.5/np.log(10))*corrected_flux_e/corrected_flux

            all_data += [[num_found,galatic_centre[0],galatic_centre[1],cal_mag,
                cal_mag_e,radius,final_area,final_flux,back_removal_e,
                expected_count_e,cutoff_correction,cutoff_correction_e,
                back_mean,back_std]]

            #add data to string line for output
            data_str = data_str + "{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n"\
                .format(num_found,*galatic_centre, radius, cal_mag, cal_mag_e)

            if num_found%100 == 0:
                print(data_str[:-1])

            num_found += 1

        #limit number of galaxies found
        if limit != None and num_found == limit:
            break

    # #commit data to txt file
    # with open(txt_name, "w+") as file:
    #     file.write(txt_str[:-1])

    with open(csv_name, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([str(search_params)])
        writer.writerow(["Name","Centre x","Centre y","Calibrated Magnitude",
        "CM Error","Radius","Area","Total Uncorrected Flux Through Aperture",
        "F Error form Bakcgorund Removal","F Number Counting Error",
        "Aperture Cutoff Flux Correction","ACFC Error","Background Flux Mean",
        "Background Flux Std"])

        writer.writerows(all_data)

    print("\n{} Galaxies Found".format(num_found))

    #show result of search
    if final_plot:
        plot_auto_search_result(csv_name)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    # tlp = (1481, 1715)
    # brp = (2033, 1263)
    # search_params = {
    #     "border" : 50,
    #     "loc_back_radius" : 50,
    #     "max_radius" : 30,
    #     "selection_std_multiplier" : 4,
    #     "growth_std_multiplier" : 2}
    tlp = (1508, 2256)
    brp = (2261, 1500)
    # tlp = (1900, 2295)
    # brp = (2261, 2000)
    search_params = {
        "border" : 50,
        "loc_back_radius" : 50,
        "max_radius" : 30,
        "selection_std_multiplier" : 4,
        "growth_std_multiplier" : 4}

    auto_search("mar_16_data.csv", tlp, brp, **search_params, final_plot = True,
        individual_plots = False)

    plot_catalogue("mar_16_data.csv")
