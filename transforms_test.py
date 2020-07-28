import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def print_pos(x,y):
    print("("+str(x)+", "+str(y)+")")
def point_str(point):
    return "("+str(point[0][0])+", "+str(point[1][0])+")"

# Transformation helpers
def get_translate_matrix(dx,dy):
    return np.array([[1,0,dx],[0,1,dy],[0,0,1]])
def get_rotate_matrix(angle): # assuming angle in degrees
    theta = angle*np.pi/180
    return np.array([[np.cos(theta), np.sin(theta),0],[-np.sin(theta), np.cos(theta),0],[0,0,1]])
def get_scale_matrix(scale_factor):
    return np.array([[scale_factor,0,0],[0,scale_factor,0],[0,0,1]])
def get_horiz_shear_matrix(slope):
    return np.array([[1,slope,0],[0,1,0],[0,0,1]])

def get_point_vector(x,y):
    return np.array([[x],[y],[1]])
def get_point_pair(point):
    return [point[0][0], point[1][0]]
def plot_point(point):
    plt.plot(point[0][0],point[1][0],'.',markersize=10)

pointA = get_point_vector(1.2,3.5)
pointB = get_point_vector(2.2,3.5)

plt.plot(pointA[0][0],pointA[1][0],'k.')
plt.xlim([0,5])
plt.ylim([0,5])

pointA1 = np.matmul(get_translate_matrix(1,1), pointA)
pointB1 = np.matmul(get_translate_matrix(1,1), pointB)

plot_point(pointA1)
plot_point(pointB1)

plt.grid()

plt.show()

