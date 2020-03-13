import matplotlib.pyplot as plt
import numpy as np

total_area = np.array([1], dtype = np.int32)
radius_range = np.arange(0,25)

for radius in radius_range[1:]:

    central_disp = np.zeros([radius*2+1, radius*2+1],dtype = np.float64)
    for i in range(radius*2+1):
        for j in range(radius*2+1):
            central_disp[i,j] =\
                np.sqrt((i-radius)**2+(j-radius)**2)
    central_disp = central_disp.flatten()
    mask = np.array([True if r <= radius else False for r in central_disp])

    total_area = np.append(total_area, len(mask[mask]))

d_total_area = total_area[1:] - total_area[:-1]
total_area_mids = (total_area[1:] + total_area[:-1])/2

fig = plt.figure()
axis = fig.add_subplot(1,1,1)

for i in range(len(radius_range)):
    axis.plot([total_area[i],total_area[i]], [-1.2,+1.2], "k", alpha = 0.5)

flat_line = np.zeros(len(total_area_mids))

axis.plot(total_area_mids, flat_line, label = "Background Mean")
axis.plot(total_area_mids, flat_line + 1, label = "Background Std")
axis.plot(total_area_mids, flat_line - 1, )

axis.plot(total_area_mids,(flat_line+1)/np.sqrt(d_total_area),
    label="Background Std")
axis.plot(total_area_mids,  - (flat_line+1)/np.sqrt(d_total_area))

axis.set_xlim(0, np.max(total_area))
axis.set_ylim(-1.2,+1.2)
axis.set_xlabel("Aperture Area")
axis.set_ylabel("Chage in Aperture Flux / Change in Aperture Area (normalised)")
plt.show()
