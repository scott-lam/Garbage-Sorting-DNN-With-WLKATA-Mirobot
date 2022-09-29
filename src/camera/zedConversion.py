from fcntl import F_EXLCK
import math

#conf settings for camera found in SNxxxxxxxx.conf (x = serial no)
fx=267.3625
fy=267.38
cx=338.76
cy=194.483

def convert_to_x_y(u, v, z):
    x = ((u - cx ) * z ) / fx 
    y = ((v - cy ) * z ) / fy
    
    return x, y