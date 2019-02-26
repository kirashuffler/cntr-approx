from __future__ import division
import cv2
import numpy as np
import itertools
import sys
import math


change_x = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
change_y = np.array([-1, -1, -1, 0, 1, 1, 1, 0])
sys.setrecursionlimit(1500)

def scanBorder_recc(img, start, prev_dir, res, points, base):
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
                base.append(len(res))
            res.append(direction)
            points.append([y, x])
            break
        #elif next_pt_value == 1:
        #    img[y][x] = 0
        #    direction = i
        #    break
    if direction == -1:

        return

    y += change_y[direction]
    x += change_x[direction]
    scanBorder_recc(img, [y, x], direction, res, points, base)


def scanBorder(img, start):
    ls = line_start(img, start)
    res = []
    points = []
    base = []
    base.append(0)
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
    res.append(direction)
    scanBorder_recc(img, [y, x], direction, res, points, base)
    img[y][x] = 0
    uncompressed_length = sys.getsizeof(res)
    res = chain_compressor(res)
    compressed_length = sys.getsizeof(res)
    #print(uncompressed_length / compressed_length)
    return [ls, res, points, base]


def line_start(img, start):
    y = start[0]
    x = start[1]
    direction = 0
    while True:
        end = True
        dirs = [direction, (direction + 1) % 8, (direction + 7) % 8, (direction + 2) % 8,
                (direction + 6) % 8, (direction + 3) % 8, (direction + 5) % 8, (direction + 4) % 8]
        for i in dirs:
            next_pt_value = img[y + change_y[i]][x + change_x[i]]
            if next_pt_value == 255:
                end = False
                print([y, x])
                img[y][x] = 1
                direction = i
                y += change_y[direction]
                x += change_x[direction]
                break
        if end:
            print("wwo")
            return [y, x]


def find_start(img):
    start_pt = (0, 0)
    h, w = img.shape
    for j, i in itertools.product(range(1, h - 1), range(1, w - 1)):
        if img[j][i] > 0:
            start_pt = (j, i)
            break
    return start_pt


def draw_by_chain(img, start_pt, chain):
    y = start_pt[0]
    x = start_pt[1]
    #print(chain)
    for dir in chain:
        if type(dir) is int:
            img[y][x] = 255
            y += change_y[dir]
            x += change_x[dir]
        else:
            for i in range(dir[1]):
                #print(dir)
                img[y][x] = 255
                y += change_y[dir[0]]
                x += change_x[dir[0]]


def draw_all_chains(chains, h, w):
    img = np.zeros((h, w, 3), np.uint8)
    for chain in chains:
        draw_by_chain(img, chain[0], chain[1])
    return img


def chain_compressor(chain):
    chain.reverse()
    compressed = []
    previous = chain.pop()
    count = 1
    while len(chain) > 0:
        current = chain.pop()
        if current == previous:
            count += 1
        else:
            if count == 1:
                compressed.append(previous)
            else:
                compressed.append([previous, count])
            count = 1
        previous = current
    if count == 1:
        compressed.append(previous)
    else:
        compressed.append([previous, count])
    return compressed


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
                if result == None or len(result[2]) < threshold:
                    continue
                contours.append(result)
    contours.sort(key=sortByLength)
    return contours


def sortByLength(inputChain):
    return len(inputChain[2])


def printPoints(img, points):
    for point in points:
        img[point[0]][point[1]] = 255

def dominantPoints(points, base, thresh):
    length = len(base)
    base_pts = []
    for index in base:
        base_pts.append(points[index])
    pts_cos = []
    r = [] #support region length
    for i in range (length - 1):
        a = (points[base[i + 1]][0] - points[base[i]][0], points[base[i + 1]][1] - points[base[i]][1])
        b = (points[base[i - 1]][0] - points[base[i]][0], points[base[i - 1]][1] - points[base[i]][1])
        len_a = math.sqrt(a[0] * a[0] + a[1] * a[1])
        len_b = math.sqrt(b[0] * b[0] + b[1] * b[1])
        cos = (a[0] * b[0] + a[1] * b[1]) / (len_a * len_b)
        pts_cos.append(abs(cos))
        regionLength = abs(base[i + 1] - base[i - 1]) + 1
        r.append(regionLength)
    a = (points[base[0]][0] - points[base[-1]][0], points[base[0]][1] - points[base[-1]][1])
    b = (points[base[-2]][0] - points[base[-1]][0], points[base[-2]][1] - points[base[-1]][1])
    regionLength = abs(base[0] - base[-2]) + 1
    r.append(regionLength)
    cos = (a[0] * b[0] + a[1] * b[1]) / math.sqrt((a[0] * a[0] + a[1] * a[1]) * (b[0] * b[0] + b[1] * b[1]))
    pts_cos.append(abs(cos))
    print(pts_cos)
    pointsToRemove = []
    for i in range(length - 1):
        if pts_cos[i] > thresh:
            pointsToRemove.append(i)
        elif pts_cos[i] < pts_cos[i - 1] or pts_cos[i] < pts_cos[i + 1]:
            pointsToRemove.append(i)
        elif pts_cos[i] == pts_cos[i - 1] and r[i] < r[i - 1]:
            pointsToRemove.append(i)
        elif pts_cos[i] == pts_cos[i + 1] and r[i] < r[i + 1]:
            pointsToRemove.append(i)
        elif pts_cos[i] == pts_cos[i + 1] and r[i] == r[i + 1]:
            pointsToRemove.append(i)
    print(len(pointsToRemove))
    pointsToRemove.reverse()
    for ind in pointsToRemove:
        del base_pts[ind]
    return base_pts



img = cv2.imread("osu.jpg", 0)
blur = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blur, 0, 100)
#edges = cv2.imread("x.png", 0)
color = [0, 0, 0]
edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
h, w = edges.shape
cv2.imwrite('edges.png', edges)
cntr = fetch_all(edges, 50)
print(cntr[-1])
img = np.zeros((h, w, 3), np.uint8)
#draw_by_chain(img, cntr[0], cntr[1])
printPoints(img, cntr[-2][2])
img = draw_all_chains(cntr, h, w)
cv2.imwrite('border.png', img)
dom_pts = dominantPoints(cntr[-2][2], cntr[-2][3], 0.9)
img = np.zeros((h, w, 3), np.uint8)
#draw_by_chain(img, cntr[0], cntr[1])
printPoints(img, dom_pts)
cv2.imwrite('dominant.png', img)