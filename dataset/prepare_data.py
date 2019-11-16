from math import pi
from random import *
import numpy as np
import cv2

from utils.constants import Car


def make_free_space(a, dtype=np.float32):
    free_space = []
    for i in range(len(a)):
        xlb = a[i][0]
        ylb = a[i][1]
        xrt = a[i][2]
        yrt = a[i][3]
        free_space.append([[xlb, ylb], [xrt, ylb], [xrt, yrt], [xlb, yrt]])
    free_space = np.array(free_space, dtype=dtype)
    return free_space


def prostopadle(path, id=0):
    r1xlb = 1.0
    r1ylb = 0.0
    r1xrt = r1xlb + 0.5 * random() + 4.0
    r1yrt = 4 * random() + 13

    width = random() + 3.5
    r2xlb = r1xrt
    r2yrt = r1yrt - 2 * random()
    r2ylb = r2yrt - width
    r2xrt = r2xlb + 4.5 + random()

    x0 = (r1xlb + r1xrt) / 2 + 0.5 * 2 * (random() - 0.5)
    y0 = r1ylb + Car.rear_axle_to_back + 3 * random()
    th0 = pi / 2 + pi / 180 * 5 * 2 * (random() - 0.5)
    ddy0 = 0.1 * 2 * (random() - 0.5)

    xk = r2xlb + Car.rear_axle_to_back + 0.4 * random()
    yk = r2ylb + width / 2 + 0.2 * 2 * (random() - 0.5)
    thk = pi / 180 * 5 * 2 * (random() - 0.5)

    free_space = make_free_space([[r1xlb, r1ylb, r1xrt, r1yrt], [r2xlb, r2ylb, r2xrt, r2yrt]])

    # image preparing
    W = 200
    H = 200
    W_range = 20.
    H_range = 20.

    r1xlb_px = round(W * r1xlb / W_range)
    r1yrt_px = round(H * (1 - r1ylb / H_range))
    r1xrt_px = round(W * r1xrt / W_range)
    r1ylb_px = round(H * (1 - r1yrt / H_range))

    r2xlb_px = round(W * r2xlb / W_range)
    r2yrt_px = round(H * (1 - r2ylb / H_range))
    r2xrt_px = round(W * r2xrt / W_range)
    r2ylb_px = round(H * (1 - r2yrt / H_range))

    free_space_px = make_free_space(
        [[r1xlb_px, r1ylb_px, r1xrt_px, r1yrt_px], [r2xlb_px, r2ylb_px, r2xrt_px, r2yrt_px]], np.int32)
    m = np.zeros((W, H), dtype=np.uint8)
    cv2.fillPoly(m, free_space_px, 255)

    # saves
    fname = path + str(id).zfill(6)

    cv2.imwrite(fname + ".png", m)

    free_space = np.reshape(free_space, (-1, 8))
    np.savetxt(fname + ".map", free_space, fmt='%.2f', delimiter=' ')

    p0pk = np.array([[x0, y0, th0, ddy0], [xk, yk, thk, 0.0]])
    np.savetxt(fname + ".scn", p0pk, fmt='%.4f', delimiter='\t')


def tunel(path, id=0):
    W1 = 15.
    r1xlb = 0.
    r1ylb = 0.
    r1xrt = r1xlb + 2. * random() + W1
    r1yrt = r1ylb + 2 * random() + W1

    width = random() + 3.0
    r2xlb = (r1xrt - width) * random()
    r2ylb = r1yrt
    r2xrt = r2xlb + width
    r2yrt = r2ylb + 8. + 6. * random()

    r3xlb = r1xlb
    r3ylb = r2yrt
    r3xrt = r3xlb + r1xrt - r1xlb
    r3yrt = r3ylb + r1yrt - r1ylb

    x0 = (r1xlb + r1xrt) / 2 + 0.5 * 2 * (random() - 0.5)
    y0 = r1ylb + Car.rear_axle_to_back + 1 * random() + 0.1
    th0 = pi / 2 + pi / 180 * 5 * 2 * (random() - 0.5)
    ddy0 = 0.1 * 2 * (random() - 0.5)

    xk = (r3xlb + r3xrt) / 2 + 0.5 * 2 * (random() - 0.5)
    yk = r3yrt - Car.rear_axle_to_front - 1 * random() - 0.5
    thk = pi / 180 * 5 * 2 * (random() - 0.5) + pi / 2

    free_space = make_free_space([[r1xlb, r1ylb, r1xrt, r1yrt], [r2xlb, r2ylb, r2xrt, r2yrt], [r3xlb, r3ylb, r3xrt, r3yrt]])
    print(free_space)

    # image preparing
    W = 256
    H = 256
    W_range = 50.
    H_range = 50.

    r1xlb_px = round(W * r1xlb / W_range)
    r1yrt_px = round(H * (1 - r1ylb / H_range))
    r1xrt_px = round(W * r1xrt / W_range)
    r1ylb_px = round(H * (1 - r1yrt / H_range))

    r2xlb_px = round(W * r2xlb / W_range)
    r2yrt_px = round(H * (1 - r2ylb / H_range))
    r2xrt_px = round(W * r2xrt / W_range)
    r2ylb_px = round(H * (1 - r2yrt / H_range))

    r3xlb_px = round(W * r3xlb / W_range)
    r3yrt_px = round(H * (1 - r3ylb / H_range))
    r3xrt_px = round(W * r3xrt / W_range)
    r3ylb_px = round(H * (1 - r3yrt / H_range))

    free_space_px = make_free_space(
        [[r1xlb_px, r1ylb_px, r1xrt_px, r1yrt_px], [r2xlb_px, r2ylb_px, r2xrt_px, r2yrt_px], [r3xlb_px, r3ylb_px, r3xrt_px, r3yrt_px]], np.int32)
    m = np.zeros((W, H), dtype=np.uint8)
    cv2.fillPoly(m, free_space_px, 255)

    # saves
    fname = path + str(id).zfill(6)

    cv2.imwrite(fname + ".png", m)

    free_space = np.reshape(free_space, (-1, 8))
    np.savetxt(fname + ".map", free_space, fmt='%.2f', delimiter=' ')

    p0pk = np.array([[x0, y0, th0, ddy0], [xk, yk, thk, 0.0]])
    np.savetxt(fname + ".scn", p0pk, fmt='%.4f', delimiter='\t')

#path = "../../data_/train/parkowanie_prostopadle/"
#path = "../../data/val/parkowanie_prostopadle/"
#path = "../../data/train/tunel/"
#path = "../../data_/train/tunel/"
#path = "../../data/val/tunel/"
path = "../../data_/val/tunel/"
for i in range(128):
    #prostopadle(path, i)
    tunel(path, i)
