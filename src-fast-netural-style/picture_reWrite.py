import cv2 as cv

def reWriteImg(img_path, img_size):
    img = cv.imread(img_path)
    img = cv.resize(img, (img_size, img_size))
    cv.imwrite(img_path, img)

reWriteImg("src-fast-netural-style/images/output/starrynight/starry_night_duck.jpg", 512)