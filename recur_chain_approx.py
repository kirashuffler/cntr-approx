from __future__ import division
import cv2
import numpy as np
import itertools
import sys
import math
import matplotlib.pyplot as plt
import scipy.interpolate as sc

change_x = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
change_y = np.array([-1, -1, -1, 0, 1, 1, 1, 0])
sys.setrecursionlimit(1500)


def scanBorder_recc(img, start, prev_dir, points):
    dirs = [prev_dir, (prev_dir + 1) % 8, (prev_dir + 7) % 8, (prev_dir + 2) % 8,
                   (prev_dir + 6) % 8, (prev_dir + 3) % 8, (prev_dir + 5) % 8]
    y = start[0]
    x = start[1]
    direction = -1
    for i in dirs:
        next_pt_value = img[y + change_y[i]][x + change_x[i]]
        if next_pt_value:
            img[y][x] = 0
            direction = i
            if direction != prev_dir:
                points.append((y, x))
            break
        #elif next_pt_value == 1:
        #    img[y][x] = 0
        #    direction = i
        #    break
    if direction == -1:
        points.append((y, x))
        return

    y += change_y[direction]
    x += change_x[direction]
    scanBorder_recc(img, [y, x], direction, points)


def scanBorder(img, start):
    ls = line_start(img, start)
    points = []
    points.append(ls)
    y = ls[0]
    x = ls[1]
    direction = -1
    for dir in range(8):
        if img[y + change_y[dir]][x + change_x[dir]]:
            direction = dir
            break
    if direction == -1:
        return None
    y += change_y[direction]
    x += change_x[direction]
    scanBorder_recc(img, [y, x], direction, points)
    img[y][x] = 0
    return points


def line_start(img, start):
    y = start[0]
    x = start[1]
    direction = 0
    prev_dir = 0
    while True:
        prev_dir = direction
        end = True
        dirs = [direction, (direction + 1) % 8, (direction + 7) % 8, (direction + 2) % 8,
                (direction + 6) % 8, (direction + 3) % 8, (direction + 5) % 8, (direction + 4) % 8]
        for i in dirs:
            next_pt_value = img[y + change_y[i]][x + change_x[i]]
            if next_pt_value == 255:
                end = False
                img[y][x] = 1
                direction = i
                y += change_y[direction]
                x += change_x[direction]
                break
        if end:
            direction = prev_dir
            dirs = [direction, (direction + 1) % 8, (direction + 7) % 8, (direction + 2) % 8,
                    (direction + 6) % 8, (direction + 3) % 8, (direction + 5) % 8, (direction + 4) % 8]

            for i in dirs:
                ym = y
                xm = x
                ym += 2 * change_y[i]
                xm += 2 * change_x[i]
                if img[ym][xm] == 255:
                    img[ym - change_y[i]][xm - change_x[i]] = 1
                    break
            return (y, x)


def find_start(img):
    start_pt = (0, 0)
    h, w = img.shape
    for j, i in itertools.product(range(1, h - 1), range(1, w - 1)):
        if img[j][i] > 0:
            start_pt = (j, i)
            break
    return start_pt


def fetch_all(img, threshold):
    start_pt = (0, 0)
    h, w = img.shape
    contours =[]
    for j in range(h):
        for i in range(w):
            if img[j][i] == 255:
                start_pt = [j, i]
                result = scanBorder(img, start_pt)
                #print(start_pt)
                if result == None or len(result[0]) < threshold:
                    continue
                contours.append(result)
    contours.sort(key=sortByLength)
    return contours


def sortByLength(inputChain):
    return len(inputChain[0])


def drawPoints(img, points):
    for point in points:
        img[point[0]][point[1]] = 255


def drawCompressed(img, points):
    l = len(points)
    for i in range(l-1):
        cv2.line(img, (points[i][1], points[i][0]), (points[i+1][1], points[i+1][0]), (255,255,255), 1)
        img[points[i]] = (0, 0, 255)


def vectorCoords(a , b):
    return (abs(a[0] - b[0]), abs(a[1] - b[1]))


def vectorModul(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1])


def scalarMultip(a, b):
    return a[0] * b[0] + a[1] * b[1]

def distance(a, b):
    return vectorModul(vectorCoords(a,b))


def dominantPoints(points, thresh):
    is_close = False
    start = 1
    if distance(points[0], points[-1]) == 0:
        points.pop()
        is_close = True
        start = 0
    length = len(points)
    base_pts = points
    pts_cos = []
    i = start + 1
    while True:
        if points[i] == points[-2]:
            break
        a_cur = vectorCoords(points[i+1], points[i])
        b_cur = vectorCoords(points[i-1], points[i])
        a_prev = vectorCoords(points[i + 1 - 1], points[i - 1])
        b_prev = vectorCoords(points[i - 1 - 1], points[i - 1])
        a_next = vectorCoords(points[i + 1 + 1], points[i + 1])
        b_next = vectorCoords(points[i - 1 + 1], points[i + 1])
        len_a_cur = vectorModul(a_cur)
        len_b_cur = vectorModul(b_cur)
        len_a_prev = vectorModul(a_prev)
        len_b_prev = vectorModul(b_prev)
        len_a_next = vectorModul(a_next)
        len_b_next = vectorModul(b_next)
        cos_cur = scalarMultip(a_cur, b_cur) / (len_a_cur * len_b_cur)
        cos_prev = scalarMultip(a_prev, b_prev) / (len_a_prev * len_b_prev)
        cos_next = scalarMultip(a_next, b_next) / (len_a_next * len_b_next)
        region_cur = len_a_cur + len_b_cur
        region_prev = len_a_prev + len_a_prev
        region_next = len_a_next + len_b_next
        toRemove = False
        if cos_cur > thresh:
            print("case1")
            toRemove = True
        #elif cos_cur < cos_prev or cos_cur < cos_next:
        #    print("case2")
        #    toRemove = True
        elif cos_cur == cos_prev and region_cur < region_prev:
            print("case3")
            toRemove = True
        elif cos_cur == cos_next and region_cur < region_next:
            print("case4")
            toRemove = True
        elif cos_cur == cos_next and region_cur == region_next:
            print("case5")
            toRemove = True
        if toRemove:
            del base_pts[i]
            i -= 1
        i += 1
    print("compression rate:", length / len(base_pts))
    return base_pts

def sortsecond(val):
    return val[1]


img = cv2.imread("x.png", 0)
blur = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blur, 0, 100)
color = [0, 0, 0]
edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
h, w = edges.shape
cv2.imwrite('edges.png', edges)
cntr = fetch_all(edges, 0)

img = np.zeros((h, w, 3), np.uint8)
drawCompressed(img, cntr[0])
cv2.imwrite('extracted.png', img)
dominant_pts = dominantPoints(cntr[0], 1)

img = np.zeros((h, w, 3), np.uint8)
drawCompressed(img, dominant_pts)
cv2.imwrite('dominant.png', img)
dominant_pts = list(dominant_pts)
dominant_pts = np.asarray(dominant_pts)

y = dominant_pts[:, 0]
x = dominant_pts[:, 1]
y -= min(y)
x -= min(x)


# fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
# is needed in order to force the spline fit to pass through all the input points.
tck, u = sc.splprep([x, y])
print(tck[0])
# evaluate the spline fits for 1000 evenly spaced distance values
xi, yi = sc.splev(np.linspace(0, 1, 1000), tck)

# plot the result
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, 'or')
ax.plot(xi, yi, '-b')
plt.gca().invert_yaxis()
plt.show()