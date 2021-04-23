# Dokumentasi Image Processing
 Mempelajari Image processing pada openCV
## Changing Color spaces
Disini dipelajari mengenai cara mengonversi ganmbar dari suatu colorspace ke colorspace lalnnya. di OpenCV ini terdapat lebih dari 150 colorspace tetapi disini hanya dipelajari 2 color space yaitu BGR ke Gray dan BGR ke HSV untuk mengkonversi colorpsace tersebut diperlukan fungsi `cvtColor()`. Pengkonversian colorspace ini akan sangat berguna untuk object tracking. Pada object tracking ini akan dilakukan perkonversian colorspace dari BGR ke HSV lalu untuk mengambil warna yang mau di track maka memerlukan fungsi `inRange()` dan untuk menampilkan warna yang di track akan menggunakan fungsi `bitwise_and`.

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
<br>
cv.THRESH_BINARY  <br>
cv.THRESH_BINARY_INV <br>
cv.THRESH_TRUNC <br>
cv.THRESH_TOZERO <br>
cv.THRESH_TOZERO_INV <br>
## Adaptive Thresholding
Jika kita berada di suatu ruang dengan kondisi lighting yang kurang bagus adaptive thresholding ini akan membantu. Algoritma pada threshod ini <nr>
 di adaptiveThresholding ini terdapat dua metode yaitu : <br>
 cv.Adaptive_Thresh_Mean_C :  Nilai dari threshiold ini adalah rata rata dari daerah sekita di kurangkan dengan C konstan <br<
 cv.ADAPTIVE_THRESH_GAUSSIAN : nilai dari threshold ini adalah jumlah bobot dari gaussian di daerak skeita dikurangkan dengan c konstan
## Otsu's Binarization :<br>
di threshold global kita menggunakan nilai yang dipilih sebagai threshold. Metode otsu ini untuk mencegah memilih value lalu ditentukan secara otomatis. bayangkan saja jika sebuah gambar memiliki
## Smoothin Image
### 2D Convolution
gambar bisa difilter menggunakan lowpasfilter dan juga highpass filters. lpf digunakan untuk meremove Noise, Bluurring Image dll. lalu HPF berguna untuk mecari edge pada gambar <br>
### Image Blurring
 di opencv terdapat 4 macam teknik untuk memblur gambar
 #### 1.Averaging
 pada averaging ini, digunakan fungsi `cv.blur` atau `cv.boxFilter` untuk memblur gambar. cara kerja dari averaging ini adalah merata rata semua pixel yang berada di daerah kernel dan mengganti elemn di tengah.
 #### 2. Gaussian Blurring
 pada gaussiaan ini menggunnakan fungsi cv.GaussiaBlur(). di sini harus ditentukan lebar dan tinggi dari kernel yang harus positif dan ganjil. gaussian blurring ini  sangat efektif unutk meremove gaussian noise
 #### 3. Median Blurring
 Pada fungsi `cv.MedianBlur()` ini akan mengambil nilai tengah dari semuah pixel yang berada di daerah kernel lalu elemen tengahnya akan diganti dengan nilai median. Fungsi ini akan sangat efektif menghilangkan "salt and paper noise" di sebuah gambar
 
 ## Contours
 contour ini adalah alat yang berhuna untuk analisis bentuk dan deteksi serta mengenal objek objek. contour ini bisa dicari menggunakan fungsi `cv.findCOuntours` yang memili parameter sumber gambar, contour retrieval mode dan yang ketiga adalah metode aproksimasi. luas pada contour ini bisa 
 
 ##Tugas Opsional OpenCV warna dalam warna
 Pertama frame akan diconvert ke hsv
 ```
 hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 ```
 lalu dibuat `lower_blue` dan `upper_blue` untuk dimasking
 ```
 lower_blue = np.array([100,150,0])
 upper_blue = np.array([140,255,255])
 mask = cv.inRange(hsv, lower_blue, upper_blue)
 ```
 jika terdapat warna biru maka akan dicari nilai paling luas 
 ```
 blue_area = max(bluecnts, key=cv.contourArea)
 ```
 gambar rectangle dan juga textnya 
 ```
 imageFrame = cv.rectangle(frame, (x, y), 
                                (x + w, y + h), 
                                (0, 255, 0), 2)
        cv.putText(imageFrame, "Warna Biru", (x, y),
                        cv.FONT_HERSHEY_DUPLEX, 1.0,
                        (0, 255, 0))
        rect = hsv[y:y+h,x:x+w]
 ```
 oper nilai hsv dari kotak dan juga frame pada kotak ke fungsi red contours untuk mencari warna merah
 ```
 redContours(rect,imageFrame)
 ```
 cari warna merah lalu gambar kotak dan tambahkam textnya
 ```
 lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask2 = cv.inRange(rect, lower_red, upper_red)
        redcnts= cv.findContours(mask2,
                                cv.RETR_TREE,
                                cv.CHAIN_APPROX_SIMPLE)[-2]
        if redcnts :
            red_area = max(redcnts, key=cv.contourArea)
            (x, y, w, h) = cv.boundingRect(red_area)
            imageFrame = cv.rectangle(frame2, (x, y), 
                                    (x + w, y + h), 
                                    (0, 0, 255), 2)
              
            cv.putText(imageFrame, "Red Colour", (x, y),
                        cv.FONT_HERSHEY_DUPLEX, 1.0,
                        (0, 0, 255))
 ```
 tekan 'd' jika ingin mengakhiri
 ```
 if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
```
 
 


