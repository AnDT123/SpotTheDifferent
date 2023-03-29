import cv2
import numpy as np
import math
import random
import time
import sys



size = 500
places = 5
prev = 0
curr = 0
canvas = np.ones((size,size, 3), dtype=np.uint8)  * 255
canvas2 = canvas = np.ones((size,size, 3), dtype=np.uint8)  * 255
rotation_list = [0,90,180,270]

catimg = cv2.imread("cat.jpg")
dogimg = cv2.imread("dog.jpg")

img_list = [catimg, dogimg]
x = 250
y = 400
radius = 60

rotation_angle = 90

#Method to rotate original image and insert it to position x,y on a canvas image
def rotateAndPlace(image, rotation_angle,x,y,canvas):

    h, w = image.shape[:2]

    scale = 2*radius /math.sqrt(h*h + w*w)

    w = int(w*scale)
    h = int( h*scale)

    image = cv2.resize(image, (w,h))
    width = image.shape[1]

    rotation_mat = cv2.getRotationMatrix2D((w/2, h/2), -rotation_angle, 1.)

    canvas_height = canvas.shape[0]
    canvas_width = canvas.shape[1]

    rotation_mat[0, 2] += x - w/2
    rotation_mat[1, 2] += y -h/2

    rotated_image = cv2.warpAffine(image,
                                   rotation_mat,
                                   (canvas_width, canvas_height))

    rotated_image = rotated_image.astype(float)
    canvas = canvas.astype(float)

    canvas = cv2.add(rotated_image, canvas)
    canvas = canvas.astype(np.uint8)
    return canvas

# Generate random circle centers
centers = []
print("Choosing position <",end='\r')
while len(centers) < places:
    # Generate a random circle center
    center = np.random.randint(radius, size - radius, size=(2,))
    
    # Check if the circle overlaps with any existing circles
    overlaps = False
    for c in centers:
        if np.linalg.norm(c - center) < 2 * radius:
            overlaps = True
            break
    
    # If the circle doesn't overlap, add it to the list of centers
    if not overlaps:
        centers.append(center)
    curr = curr +1
    if len(centers) - prev >= int(places/20):
        print("=", end='\r')
        prev = len(centers)
print(">")
prev = 0
curr = 0
print("Drawing image     <",end='\r')
# Draw image on each center position with random rotation angle (90,180,270)
for center in centers:
    angle  = random.choice(rotation_list)
    img = random.choice(img_list)
    canvas = rotateAndPlace(img, angle,center[1],center[0],canvas)
    img = random.choice(img_list)
    canvas2 = rotateAndPlace(img, angle,center[1],center[0],canvas2)
    curr = curr +1
    if curr - prev >= int(places/20):
        print("=", end='\r')
        prev = curr
print(">")
cv2.imwrite("image1.jpg", canvas)
cv2.imwrite("image2.jpg", canvas2)
print("Done... _/(@@)\_")
