import cv2
import numpy as np
import itertools

change_x = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
change_y = np.array([-1, -1, -1, 0, 1, 1, 1, 0])


def scanBorder_recc(img, start, prev_dir, res):
    dirs = [prev_dir, (prev_dir + 1) % 8, (prev_dir + 7) % 8, (prev_dir + 2) % 8,
                   (prev_dir + 6) % 8, (prev_dir + 3) % 8, (prev_dir + 5) % 8, (prev_dir + 4) % 8]
    y = start[0]
    x = start[1]
    direction = -1

    for i in dirs:
        next_pt_value = img[y + change_y[i]][x + change_x[i]]
        if next_pt_value:
            img[y][x] = 0
            direction = i
            res.append(direction)

            break
        #elif next_pt_value == 1:
        #    img[y][x] = 0
        #    direction = i
        #    break
    if direction == -1:
        return

    y += change_y[direction]
    x += change_x[direction]
    scanBorder_recc(img, [y, x], direction, res)


def scanBorder(img, start):
    ls = line_start(img, start)
    print(ls)
    res = []
    y = ls[0]
    x = ls[1]
    direction = -1
    for dir in range(8):
        if img[y + change_y[dir]][x + change_x[dir]]:
            direction = dir
            break
    y += change_y[direction]
    x += change_x[direction]
    res.append(direction)
    scanBorder_recc(img, [y, x], direction, res)
    img[y][x] = 0
    return [ls, res]

def line_start(img, start):
    y = start[0]
    x = start[1]
    direction = -1
    while True:
        dirs = [direction, (direction + 1) % 8, (direction + 7) % 8, (direction + 2) % 8,
                (direction + 6) % 8, (direction + 3) % 8, (direction + 5) % 8, (direction + 4) % 8]
        for i in dirs:
            next_pt_value = img[y + change_y[i]][x + change_x[i]]
            if next_pt_value == 255:
                print([y, x])
                img[y][x] = 1
                direction = i
                y += change_y[direction]
                x += change_x[direction]
                break
            elif next_pt_value == 1:
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
        #print(dir)
        img[y][x] = 255
        y += change_y[dir]
        x += change_x[dir]

img = cv2.imread("x.png", 0)
blur = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blur, 0, 100)
#edges = cv2.imread("x.png", 0)
color = [0, 0, 0]
edges = cv2.imread("x.png", 0)
edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
h, w = edges.shape
cv2.imwrite('edges.png', edges)

contour = np.zeros((h, w, 3), np.uint8)
start = find_start(img)
codes = scanBorder(img, start)
print(codes)
draw_by_chain(contour, codes[0], codes[1])
cv2.imwrite("cntr.png", contour)