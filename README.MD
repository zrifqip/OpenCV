# Dokumentasi Image Processing
 Mempelajari Image processing pada openCV
## Changing Color spaces
Disini dipelajari mengenai cara mengonversi ganmbar dari suatu colorspace ke colorspace lalnnya. di OpenCV ini terdapat lebih dari 150 colorspace tetapi disini hanya dipelajari 2 color space yaitu BGR ke Gray dan BGR ke HSV untuk mengkonversi colorpsace tersebut diperlukan fungsi `cvtColor()`. Pengkonversian colorspace ini akan sangat berguna untuk object tracking. Pada object tracking ini akan dilakukan perkonversian colorspace dari BGR ke HSV lalu untuk mengambil warna yang mau di track maka memerlukan fungsi `inRange()` dan untuk menampilkan warna yang di track akan menggunakan fungsi `bitwise_and` Implementasinya sebagai berikut :

```
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
```
## Transformation

Transformasi pada opencv ini terdapat 2 fungsi yaitu ada `warpAffine()` yang dimana digunakan untuk transformasi matrix 2 x 3 lalu ada `warpPerspective()` untuk transformasi matrix 3 x 3

### Scaling
Scaling ini digunakan untuk mengubah ukuran suatu gambar menggunakan fungsi `resize()` meresize gambar ini dapat dilakukan secara manual atau bisa menggunakan `cv.INTER_AREA` untuk mengecilkan gambar dan `cv.INTER_LINEAR` untuk memperbesar

### Translation
Translasi ini digunakan untuk menggeser gambar dari posisi awalnya

### Rotation
Rotasi ini digunakan untuk memutar gambar, gambar akan terputar sesuai dengan sudut dari input yang telah ditentukan

## Image Thresholding
Thresholding adalah sebuah teknik segmentasi untuk memisahkan objek dari background di proses dari thresholding ini akan dibandingkan setiap pixel gambar dengan nilai thresholder pada batas yang telah ditentukan. pada fungsi threshold ini terdapat 3 argumen. yang pertama adalah sumber dari gambar. lalu yang kedua adalah pixel value, dan yang ketiga adalah nilai maximum dari threshold. di paramenter keempat dari tipe thresholding yang digunakan. tipe tuipe dari threshoolding adalah sebagai berikut : <br>
cv.THRESH_BINARY  <br>
cv.THRESH_BINARY_INV <br>
cv.THRESH_TRUNC <br>
cv.THRESH_TOZERO <br>
cv.THRESH_TOZERO_INV <br>
## Smoothin Image


