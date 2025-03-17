import cv2 as cv

def reWriteImg(img_path, img_size):
    img = cv.imread(img_path)
    img = cv.resize(img, (img_size, img_size))
    cv.imwrite(img_path, img)

reWriteImg("images/Vangoh.jpg", 500)