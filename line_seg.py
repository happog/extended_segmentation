import numpy as np
import cv2
np.set_printoptions(threshold=np.nan)


def line_seg():
    im = cv2.imread('images/text.png')
    im = cv2.resize(im,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    height = im.shape[0]
    width = im.shape[1]
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    kernel = np.ones((5,5),np.uint8)
    blob = cv2.dilate(thresh,kernel,iterations = 1)
    lcount = np.zeros(shape=(height,))
    for y in range(height):
        for x in range(width):
            if blob[y][x] == 255:
                lcount[y] += 1

    #print(lcount)
    line_seg = []
    for z in range(len(lcount)):
        if z>=2 and z <= len(lcount)-3:
            if lcount[z] == 0 and max(lcount[z-2:z+3]) == 0:
                line_seg.append(z)

    #print(line_seg)
    rem_list = []
    final_seg = []
    for z in range(len(line_seg)):
        if z > 1 and z < len(line_seg)-1:
            if line_seg[z] == line_seg[z-1] +1:
                rem_list.append(line_seg[z-1])
            elif line_seg[z] != line_seg[z-1] +1 and line_seg[z-1] == line_seg[z-2] +1 and line_seg[z+1] == line_seg[z]+1:
                rem_list.append(line_seg[z-1])
                avg = int(sum(rem_list)/len(rem_list))
                final_seg.append(avg)
                rem_list = []
            elif line_seg[z] != line_seg[z-1] +1 and line_seg[z-1] == line_seg[z-2] +1 and line_seg[z+1] != line_seg[z]+1:
                rem_list.append(line_seg[z - 1])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
                final_seg.append(line_seg[z])
            elif line_seg[z] != line_seg[z-1] +1 and line_seg[z-1] != line_seg[z-2] +1 and line_seg[z+1] == line_seg[z]+1:
                rem_list.append(line_seg[z])
            else:
                final_seg.append(line_seg[z])
        elif z == 0:
            if line_seg[z] == line_seg[z+1] -1:
                rem_list.append(line_seg[z])
            else:
                final_seg.append(line_seg[z])
        elif z == 1:
            if line_seg[z] == line_seg[z-1] + 1 and line_seg[z] != line_seg[z+1] - 1:
                rem_list.append(line_seg[z])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
            elif line_seg[z] != line_seg[z-1] + 1 and line_seg[z] != line_seg[z+1] - 1:
                final_seg.append(line_seg[z])
        elif z == len(line_seg) -1:
            if line_seg[z] != line_seg[z-1] + 1:
                final_seg.append(line_seg[z])
            elif line_seg[z] == line_seg[z-1] + 1 and line_seg[z-1] != line_seg[z-2] + 1:
                rem_list.append(line_seg[z-1])
                rem_list.append(line_seg[z])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
            elif line_seg[z] == line_seg[z-1] + 1 and line_seg[z-1] == line_seg[z-2] + 1:
                rem_list.append(line_seg[z - 1])
                rem_list.append(line_seg[z])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
    return final_seg

