# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import os

print("[INFO] loading network...")
model = load_model('characters.model')


def img_preference(img):
    sliced = img[:-4]
    number = int(sliced)
    return number


file = open('output.txt','w')
# load the image
for imageitem in sorted(os.listdir('data'),key=img_preference):
    image = cv2.imread('data' + '/' + imageitem,0)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)



    # classify the input image
    #(digit0, digit1, digit2,digit3,digit4,digit5,digit6,digit7,digit8,digit9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z) = model.predict(image)[0]
    label = model.predict(image)[0]
    # build the label
    '''label = "Santa" if santa > notSanta else "Not Santa"
    proba = santa if santa > notSanta else notSanta
    label = "{}: {:.2f}%".format(label, proba * 100)'''

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    #cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    maxi = max(label)
    print(imageitem)
    print(maxi)
    index = label.tolist().index(maxi)
    real_classes = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f','g','h','i','j','k','l','m',
                    'n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E',
                    'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W',
                    'X','Y','Z']
    label_list = label.tolist()
    print(real_classes[index])
    del label_list[index]
    maxi2 = max(label_list)
    print(maxi2)
    index2 = label_list.index(maxi2)
    print(real_classes[index2])
