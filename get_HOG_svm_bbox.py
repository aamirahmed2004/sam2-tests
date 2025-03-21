import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread("sample_frames3/1035_24.jpg")
h,w = image.shape[:2]
desired_height, desired_width = 200,200

# Pad the image to 128x644
if h < desired_height or w < desired_width:
    top = max(0, (desired_height - h) // 2)
    bottom = max(0, desired_height - h - top)
    left = max(0, (desired_width - w) // 2)
    right = max(0, desired_width - w - left)
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', img_gray)
cv2.waitKey(0)

rects, weights = hog.detectMultiScale(img_gray, winStride=(2, 2), padding=(2, 2), scale=1.02)

for i, (x, y, w, h) in enumerate(rects):
    if weights[i] < 0.13:
        continue
    elif weights[i] < 0.3 and weights[i] > 0.13:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    if weights[i] < 0.7 and weights[i] > 0.3:
        cv2.rectangle(image, (x, y), (x+w, y+h), (50, 122, 255), 2)
    if weights[i] > 0.7:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('HOG detection', image)
cv2.waitKey(0)