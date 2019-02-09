import cv2
import numpy as np
import itertools

change_x = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
change_y = np.array([-1, -1, -1, 0, 1, 1, 1, 0])

def find_start(img):
    start_pt = (0, 0)
    h, w = img.shape
    for j, i in itertools.product(range(1, h - 1), range(1, w - 1)):
        if img[j][i] > 0:
            start_pt = (j, i)
            break
    return start_pt

def fetch_all(img,threshold):
    start_pt = (0, 0)
    h, w = img.shape
    contours =[]
    for j, i in itertools.product(range(1, h - 1), range(1, w - 1)):
        if img[j][i] == 255:
            start_pt = (j, i)
            #print(start_pt)
            chain = chain_algo(img, start_pt)
            if len(chain) > threshold:
                contours.append([len(chain), start_pt, chain])
            delete_by_chain(img, h, w, start_pt, chain)
    contours.sort()
    return contours

def chain_algo(img, start_pt):
    border = []
    chain = []
    y = start_pt[0]
    x = start_pt[1]
    for i in range(8):
        if img[y + change_y[i]][x + change_x[i]] == 255:
            chain.append(i)
            y += change_y[i]
            x += change_x[i]
            border.append((y, x))
            break
    while (y, x) != start_pt:
        b_dir = (i + 5) % 8
        dirs_1 = range(b_dir, 8)
        dirs_2 = range(0, b_dir)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for i in dirs:
            if img[y + change_y[i]][x + change_x[i]]:
                chain.append(i)
                y += change_y[i]
                x += change_x[i]
                border.append((y, x))
                break
        #if i == (chain[-1] + 4) % 8:
         #   break
    #print(border)
    return chain

def draw_by_border(border, h, w):
    img = np.zeros((h, w, 3), np.uint8)
    for point in border:
        img[point] = 255
    return img

def draw_by_chain(img, chain):
    y = chain[1][0]
    x = chain[1][1]
    codes = chain[2]
    #print(chain)
    for dir in codes:
        #print(dir)
        img[y][x] = 255
        y += change_y[dir]
        x += change_x[dir]

def draw_chains(chains, h, w):
    img = np.zeros((h, w, 3), np.uint8)
    for chain in chains:
        draw_by_chain(img, chain)
    return img


def delete_by_chain(img, h, w, start_pt, chain):
    y = start_pt[0]
    x = start_pt[1]
    for dir in chain:
        img[y][x] = 0
        y += change_y[dir]
        x += change_x[dir]

def size_of_chains(chains):
    s = 0
    for chain in chains:
        s += 2
        s += len(chain)
    return s



img = cv2.imread("test1.JPG", 0)
blur = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blur, 0, 100)
#edges = cv2.imread("x.png", 0)
color = [0, 0, 0]
edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
h, w = edges.shape
cv2.imwrite('edges.png', edges)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
opencv_cntrs = np.zeros((h, w, 3), np.uint8)
cv2.drawContours(opencv_cntrs, contours, -1, (255,255,255), 1)
cv2.imwrite("opencv.png", opencv_cntrs)

chains = fetch_all(edges, 100)
print(chains)

print(h * w / size_of_chains(chains))

border_img = draw_chains(chains, h, w)
cv2.imwrite("border.png", border_img)
single_line = np.zeros((h, w, 3), np.uint8)
draw_by_chain(single_line, chains[-1])
cv2.imwrite("single_line.png", single_line)
