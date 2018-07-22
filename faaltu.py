import numpy as np
import cv2
import line_seg


#im = cv2.imread('images/text.png')
im1 = cv2.imread('images/text.png')
im1 = cv2.resize(im1,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
#im = cv2.resize(im,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
#im = im[19:110,:]
#lines = line_seg.line_seg()
j = 0

im = im1[19:110]
height = im.shape[0]
width = im.shape[1]
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
thresh_real = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#thresh_inv = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
kernel = np.ones((5,5),np.uint8)
blob = cv2.dilate(thresh,kernel,iterations = 4)
blob = cv2.erode(blob,kernel,iterations=4)

count = np.zeros(shape=(height,))
for y in range(height):
    for x in range(width):
        if blob[y][x] == 255:
            count[y] = count[y] + 1
print(count)
maxy = max(count)
for z in range(len(count)):
    if z >= 2 and z <= height - 2:
        if count[z] > count[z+1] + 0.2*width and abs(count[z] - count[z-1]) < 0.2*width:
            base = z
        if count[z+1] < 10 and count[z] >= 10:
            desc = z
        if count[z] > count[z-1] + 0.2*width and abs(count[z] - count[z+1]) < 0.2*width:
            mean = z
        if count[z-1] < 10 and count[z] >= 10:
            asc = z


print(base)
print(desc)
print(mean)
print(asc)
blob1 = blob.copy()
cv2.line(blob1,(0,base),(width,base),(255,0,0),1)
cv2.line(blob1,(0,desc),(width,desc),(255,0,0),1)
cv2.line(blob1,(0,mean),(width,mean),(255,0,0),1)
cv2.line(blob1,(0,asc),(width,asc),(255,0,0),1)
blobless = cv2.dilate(thresh,kernel,iterations = 3)
count_y = []
word_image,contours,hierarchy = cv2.findContours(blobless,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

word_k = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > 20:
        word_k.append((x, y, w, h))

word = sorted(word_k, key=lambda student: student[0])
word_final = []
for e in range(len(word)):
    x = word[e][0]
    y = word[e][1]
    w = word[e][2]
    h = word[e][3]
    if e > 0:
        if word[e-1][0] <= x <= word[e-1][0]+word[e-1][2] and word[e-1][0] <= x+w <= word[e-1][0]+word[e-1][2]:
            continue
        else:
            word_final.append(word[e])
            im = cv2.rectangle(imgray, (x, y), (x + w, y + h), (0, 255, 0), 1)
    else:
        word_final.append(word[e])
        im = cv2.rectangle(imgray, (x, y), (x + w, y + h), (0, 255, 0), 1)
print(word_final)

thresh1 = thresh.copy()
for a in range(len(word_final)):
    word1 = thresh1[word_final[a][1]:word_final[a][1]+word_final[a][3],word_final[a][0]:word_final[a][0]+word_final[a][2]]
    word1_real = thresh_real[word_final[a][1]:word_final[a][1]+word_final[a][3],word_final[a][0]:word_final[a][0]+word_final[a][2]]
    ylower = mean - word_final[a][1]
    yupper = base - word_final[a][1]
    count_seg = np.zeros(shape=(word1.shape[1],))
    detail_seg = []
    if word1.shape[0] > yupper-ylower:
        for x in range(word1.shape[1]):
            for y in range(ylower,yupper+1):
                if word1[y][x] == 255:
                    count_seg[x] += 1
                    detail_seg.append((x,y))
    print(count_seg)
    print(detail_seg)
    print(word1.shape)
    print(ylower)
    print(yupper)
    pending_seg = []
    for z in range(len(count_seg)):
        if z > 0 and z < len(count_seg)-1:
            if count_seg[z] == 0 and min(count_seg[z-1],count_seg[z+1]) == 0:
                pending_seg.append(z)
            elif count_seg[z] == 0 and min(count_seg[z-1],count_seg[z+1]) > 0:
                print('confusion_zone'+str(z))
                pending_seg.append(z)
    print(pending_seg)
    rem_list = []
    final_seg = []
    for z in range(len(pending_seg)):
        if z > 1 and z < len(pending_seg)-1:
            if pending_seg[z] == pending_seg[z-1] +1:
                rem_list.append(pending_seg[z-1])
            elif pending_seg[z] != pending_seg[z-1] +1 and pending_seg[z-1] == pending_seg[z-2] +1 and pending_seg[z+1] == pending_seg[z]+1:
                rem_list.append(pending_seg[z-1])
                avg = int(sum(rem_list)/len(rem_list))
                final_seg.append(avg)
                rem_list = []
            elif pending_seg[z] != pending_seg[z-1] +1 and pending_seg[z-1] == pending_seg[z-2] +1 and pending_seg[z+1] != pending_seg[z]+1:
                rem_list.append(pending_seg[z - 1])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
                final_seg.append(pending_seg[z])
            elif pending_seg[z] != pending_seg[z-1] +1 and pending_seg[z-1] != pending_seg[z-2] +1 and pending_seg[z+1] == pending_seg[z]+1:
                rem_list.append(pending_seg[z])
            else:
                final_seg.append(pending_seg[z])
        elif z == 0:
            if pending_seg[z] == pending_seg[z+1] -1:
                rem_list.append(pending_seg[z])
            else:
                final_seg.append(pending_seg[z])
        elif z == 1:
            if pending_seg[z] == pending_seg[z-1] + 1 and pending_seg[z] != pending_seg[z+1] - 1:
                rem_list.append(pending_seg[z])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
            elif pending_seg[z] != pending_seg[z-1] + 1 and pending_seg[z] != pending_seg[z+1] - 1:
                final_seg.append(pending_seg[z])
        elif z == len(pending_seg) -1:
            if pending_seg[z] != pending_seg[z-1] + 1:
                final_seg.append(pending_seg[z])
            elif pending_seg[z] == pending_seg[z-1] + 1 and pending_seg[z-1] != pending_seg[z-2] + 1:
                rem_list.append(pending_seg[z-1])
                rem_list.append(pending_seg[z])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
            elif pending_seg[z] == pending_seg[z-1] + 1 and pending_seg[z-1] == pending_seg[z-2] + 1:
                rem_list.append(pending_seg[z - 1])
                rem_list.append(pending_seg[z])
                avg = int(sum(rem_list) / len(rem_list))
                final_seg.append(avg)
                rem_list = []
    print(final_seg)
    diff_list = []
    for k in range(len(final_seg)):
        if k > 0:
            diff = final_seg[k] - final_seg[k-1]
            diff_list.append(diff)
    diff_avg = int(sum(diff_list)/len(diff_list))
    print(diff_avg)
    for k in range(len(final_seg)):
        if k > 0:
            if final_seg[k]-final_seg[k-1] > 1.5*diff_avg:
                print(final_seg[k])
                print(count_seg[final_seg[k-1]:final_seg[k]])

    for k in range(len(final_seg)):
        cv2.line(word1, (final_seg[k], 0), (final_seg[k], word1.shape[0]), (255, 0, 0), 1)
    cv2.imshow('img',word1)
    cv2.waitKey(0)


