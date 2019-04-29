from __future__ import division
import cv2
import numpy as np
import itertools
import sys
import math
import matplotlib.pyplot as plt
import scipy.interpolate as sc
from copy import deepcopy

change_x = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
change_y = np.array([-1, -1, -1, 0, 1, 1, 1, 0])
sys.setrecursionlimit(1500)


def scan_border_recc(img, start, prev_dir, points, cntr_len):
    cntr_len[0] += 1
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
    scan_border_recc(img, [y, x], direction, points, cntr_len)


def scan_border(img, start):
    ls = line_start(img, start)
    cntr_len = []
    cntr_len.append(1)
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

    scan_border_recc(img, [y, x], direction, points, cntr_len)
    img[y][x] = 0

    return points, cntr_len[0]


def line_start(img, start):
    y = start[0]
    x = start[1]
    direction = 0
    prev_dir = 0
    while True:
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
                result = scan_border(img, start_pt)
                #print(start_pt)
                if result == None or len(result[0][0]) < threshold:
                    continue
                contours.append(result)
    contours.sort(key=sort_bylength)
    return contours


def sort_bylength(inputChain):
    return len(inputChain[0][0])

def vector_coords(a, b):
    return (abs(a[0] - b[0]), abs(a[1] - b[1]))


def vector_modul(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1])


def scalar_multip(a, b):
    return a[0] * b[0] + a[1] * b[1]

def distance(a, b):
    return vector_modul(vector_coords(a, b))


def thresholder(a, b, c):
    side1 = distance(a, b)
    side2 = distance(b, c)
    side3 = distance(a, c)
    p = side1 + side2 + side3
    p /= 2
    return (p * (p - side1) * (p - side2) * (p - side3)) / side3 / side3


def dominant_points_triangle(points, thresh):
    is_close = False
    start = 1
    base_pts = points.copy()
    if distance(base_pts[0], base_pts[-1]) == 0:
        base_pts.pop()
        is_close = True
        start = 0

    i = start + 1
    while True:
        if base_pts[i] == base_pts[-2]:
            break
        h = thresholder(base_pts[i - 1], base_pts[i], base_pts[i + 1])
        if h < thresh:
            del base_pts[i]
            i -= 1
        i += 1
    return [base_pts, is_close]


def dominant_points(points, thresh):
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
        a_cur = vector_coords(points[i + 1], points[i])
        b_cur = vector_coords(points[i - 1], points[i])
        a_prev = vector_coords(points[i + 1 - 1], points[i - 1])
        b_prev = vector_coords(points[i - 1 - 1], points[i - 1])
        a_next = vector_coords(points[i + 1 + 1], points[i + 1])
        b_next = vector_coords(points[i - 1 + 1], points[i + 1])
        len_a_cur = vector_modul(a_cur)
        len_b_cur = vector_modul(b_cur)
        len_a_prev = vector_modul(a_prev)
        len_b_prev = vector_modul(b_prev)
        len_a_next = vector_modul(a_next)
        len_b_next = vector_modul(b_next)
        cos_cur = scalar_multip(a_cur, b_cur) / (len_a_cur * len_b_cur)
        cos_prev = scalar_multip(a_prev, b_prev) / (len_a_prev * len_b_prev)
        cos_next = scalar_multip(a_next, b_next) / (len_a_next * len_b_next)
        region_cur = len_a_cur + len_b_cur
        region_prev = len_a_prev + len_a_prev
        region_next = len_a_next + len_b_next
        toRemove = False
        if cos_cur > thresh:

            toRemove = True
        elif cos_cur == cos_prev and region_cur < region_prev:
            toRemove = True
        elif cos_cur == cos_next and region_cur < region_next:
            toRemove = True
        elif cos_cur == cos_next and region_cur == region_next:
            toRemove = True
        if toRemove:
            del base_pts[i]
            i -= 1
        i += 1
    return [base_pts, is_close]


def sort_second(val):
    return val[1]


def spline_approx(cntr, th_triangle, degree_th):
    [dominant_pts, is_closed] = dominant_points_triangle(cntr[0], th_triangle)
    dominant_pts = list(dominant_pts)
    dominant_pts = np.asarray(dominant_pts)
    y = dominant_pts[:, 0]
    x = dominant_pts[:, 1]
    original = np.asarray(cntr[0])
    original_y = original[:, 0]
    original_x = original[:, 1]
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    degree = 3
    relation = len(x) / cntr[1]
    print("total cntr length compression: ", relation)
    if (relation) < degree_th:
        degree = 1
    tck, u = sc.splprep([x, y], per=is_closed, k=degree)
    #tck_copy = deepcopy(tck)
    knots = sc.splev(tck[0], tck)
    test_pt = dominant_pts[10]
    index = knots_interval(knots, test_pt)
    start = index - 3
    end = start + 3
    #tck_modify(tck_copy, start, end)
    #print("knots:", knots)
    #print("coeffs:", tck[1])
    #print("number of pts: ", len(x))
    #print("number of knots: ", len(tck[0]))
    #print("number of koeffs: ", len(tck[1][0]))
    #print("compression rate of dominant points to spline:", sys.getsizeof(dominant_pts) / sys.getsizeof(tck))
    # evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = sc.splev(np.linspace(0, 1, len(original_x)), tck)
    #_, [ex, ey] = sc.splev(np.linspace(tck[0][start+3], tck[0][end+1], 20), tck_copy)
    ex, ey = slice_of_spline(tck, start)
    plt.ylim(0, h)
    plt.xlim(0, w)
    plt.scatter(knots[0], knots[1], marker='x',color='red', label=('knots'))
    plt.scatter(test_pt[1], test_pt[0], marker='x', color='black', label=('test_pt'))
    #plt.scatter(x, y, marker='*',color='black', label=('Доминантные точки'+'(thresh='+str(th_triangle)+')'))
    plt.plot(original_x, original_y, color='green', label=('Изначальный контур'))
    # ax.plot(tck[1][0], tck[1][1], "-or")
    plt.plot(xi, yi,linestyle='--', color='purple', label=('B-сплайн степени '+str(degree)+'(thresh=')+str(degree_th)+')')
    plt.plot(ex, ey, color='red', label=('part'))
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig('cntr_'+str(th_triangle)+'_'+str(degree_th)+'.png')
    plt.show()
    return tck


def tck_modify(tck, start, end):
    koefs = tck[1].copy()

    length = len(koefs[0])
    s = 0
    e = length
    for i in range(s, start):
        koefs[0][i] = koefs[1][i] = 0
    for i in range(end+1, e):
        koefs[0][i] = koefs[1][i] = 0
    tck[1][1] = koefs


def knots_interval(knots, point):
    L = len(knots[0])
    x_k = knots[0]
    y_k = knots[1]
    x = point[1]
    y = point[0]
    for i in range(3, L-1):
        left_x = x_k[i]
        right_x = x_k[i + 1]
        left_y = y_k[i]
        right_y = y_k[i + 1]
        x_in = in_interval(x, left_x, right_x)
        y_in = in_interval(y, left_y, right_y)
        if x_in and y_in:
            return i
    return 0


def in_interval(val, left, right):
    return (left - 5 <= val < right + 5 or left + 5 >= val >= right - 5)


def slice_of_spline(tck, start):
    Kx = tck[1][0]
    Ky = tck[1][1]
    knots = tck[0]
    l = start + 3
    arr = np.linspace(knots[l],knots[l+1], 20)
    print(l)
    res_x = []
    res_y = []
    for point in arr:
        buff_x = [0] * (3 + 1)
        buff_y = [0] * (3 + 1)
        for i in range(3 + 1):
            buff_x[i] = Kx[i + l - 3]
            buff_y[i] = Ky[i + l - 3]
        for j in range(1, 3 + 1):
            for i in reversed(range(l - 3 + j, l + 1)):
                alpha = (point - knots[i]) / (knots[i + 1 + 3 - j] - knots[i])
                buff_x[i - l + 3] = alpha * buff_x[i - l + 3] + (1 - alpha) * buff_x[i - 1 - l + 3]
                buff_y[i - l + 3] = alpha * buff_y[i - l + 3] + (1 - alpha) * buff_y[i - 1 - l + 3]
        res_x.append(buff_x[3])
        res_y.append(buff_y[3])
    return res_x, res_y




img = cv2.imread("x.png", 0)
blur = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blur, 0, 100)
color = [0, 0, 0]
edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
#edges = cv2.imread("x.png", 0)
h, w = edges.shape
cv2.imwrite('edges.png', edges)
cntr = fetch_all(edges, 2)
spline_approx(cntr[0], 0.4, 0.01)


