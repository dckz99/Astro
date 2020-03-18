"""
Graphical tests of the behaviour of non cicular apertures.
"""

import matplotlib.pyplot as plt
import numpy as np

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

# c_r = (255,0,0)
# c_o = (255,127,0)
# c_y = (255,255,0)
# c_g = (0,255,0)
# c_b = (0,0,255)
# c_i = (75,0,130)
# c_v = (148,0,211)
# colours = [c_r,c_o,c_y,c_g,c_b,c_i,c_v]

c_r = (255,154,162)
c_o = (255,183,178)
c_y = (255,218,193)
c_g = (226,240,203)
c_b = (181,234,215)
c_i = (199,206,234)
colours = [c_r,c_o,c_y,c_g,c_b,c_i]

for i in range(len(colours)):
    c = colours[i]
    colours[i] = (c[0]/255,c[1]/255,c[2]/255)

radius = 11
image_width = radius*2+1
image = np.zeros([image_width,image_width,4], dtype = np.float64)
central_displacement_master = central_displacement_map(radius)

for r in np.arange(0,radius+1)[::-1]:
    for i in range(image_width):
        for j in range(image_width):
            if central_displacement_master[i,j] <= r:
                image[i,j,2] = colours[r%len(colours)][2]
                image[i,j,1] = colours[r%len(colours)][1]
                image[i,j,0] = colours[r%len(colours)][0]
                image[i,j,3] = 1

plt.imshow(image)
plt.xlabel("j")
plt.ylabel("i")

plt.plot([radius-.5,radius-.5],[radius+.5,-0.5],"--k",alpha = 0.7)
plt.plot([radius-.5,image_width-0.5],[radius-.5,radius-.5],"--k",alpha = 0.7)
plt.plot([radius+.5,-0.5],[radius+.5,radius+.5],"--k",alpha = 0.7)
plt.plot([radius+.5,radius+.5],[radius-.5,image_width-0.5],"--k",alpha = 0.7)

radi = [3,5,2,4]
num = 240
x = np.zeros(num)
y = np.zeros(num)
theta = np.linspace(0,np.pi*2,num = num)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

for quad in [0,1,2,3]:
    index = (int(num/4*quad), int(num/4*(quad+1)))
    x[index[0]:index[1]] = cos_theta[index[0]:index[1]] * radi[quad] + radius
    y[index[0]:index[1]] =-sin_theta[index[0]:index[1]] * radi[quad] + radius
x = np.append(x, x[0])
y = np.append(y, y[0])

plt.plot(x,y)
plt.show()
