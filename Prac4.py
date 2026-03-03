import cv2

# read image
image = cv2.imread("image3.jpg")

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)

# rotate image by 45 degrees
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated Image", rotated)

#Resize image
resized= cv2.resize(image,(200,200))
cv2.imshow("resize image",resized)

#Cropping the image
crop=image[50:200,100:300]
cv2.imshow("Croped image",crop)

#blur the image
blur= cv2.GaussianBlur(image,(7,7),0)
cv2.imshow("Blurred image",blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
