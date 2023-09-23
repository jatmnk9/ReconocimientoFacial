from PIL import Image
import numpy as np
import cv2

prueba = 'kevin.png'

def menu():
    menu = """
    ---------HOMEWORK 3: IMAGE TRANSFORMATIONS----------
    1.Mostrar canal rojo
    2.Reducir la resolución espacial a 72dpi
    3.Seleccionar canal y cambiar de la imagen a
      16 y 2 niveles de intensidad
    4.Rotar 90°
    5.Reflejar la imagen
    6.Aplicar una mascara circular al rostro
    7.Aplicar un efecto shear
    8.Aplicar filtros del 1-7
    Elija una opcion: """
    option = int(input(menu))
    if option == 1:
        img = cv2.imread(prueba)
        b, g, r = cv2.split(img)
        img[:,:,0]=0
        img[:,:,1]=0
        cv2.imshow('Red',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("canal_rojo.png", img)
    elif option ==2:
        image = cv2.imread(prueba)
        #Reducir la resolución espacial a 72dpi
        image_ = Image.fromarray(image)
        image_.info['dpi'] = (72, 72)
        im_array = np.array(image_)
        cv2.imshow('Image_72', im_array)
        cv2.imwrite("Image_72.jpg", im_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif option ==3:
        im = Image.open(prueba)
        im_array = np.array(im)
        im_array16 = (im_array // 16)*  16
        im_16 = Image.fromarray(im_array16)
        im_16.save("imagen_16.png")
        im_16.show("imagen_16.png")
        im2 = im.convert('1')
        im2.save("imagen_2.png")
        im2.show("imagen_2.png")
    elif option ==4:
        image = cv2.imread(prueba)
        image_norm = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('Rotated Image', image_norm)
        cv2.imwrite("Rotated Image.jpg", image_norm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif option ==5:
        imagen = cv2.imread(prueba)
        flip0 = cv2.flip(imagen,1)
        cv2.imshow('Reflected Image',flip0)
        cv2.imwrite("Reflected Image.jpg", flip0)
        cv2.waitKey(0)
        
    elif option ==6:
        #Reconocimiento facial
        faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(prueba)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceClassif.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            maxSize=(800,800))

        for (x,y,w,h) in faces:
            cv2.circle(img, (int(x + w/2), int(y + h/2)), int(h/2), (255, 0, 0), 2)
        
        cv2.imshow('Face detected Image',img)
        cv2.imwrite("Face detected Image.jpg", img)
        cv2.waitKey(0)

    elif option ==7:
        # Leer imagen
        img = cv2.imread(prueba)
        # Obtener las dimensiones de la imagen
        height, width = img.shape[:2]
        # Generar la matriz de transformación shear
        matrix = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)
        # Aplicar el efecto shear
        img_shear = cv2.warpAffine(img, matrix, (width, height))
        # Mostrar la imagen modificada
        cv2.imshow("Shear image", img_shear)
        cv2.imwrite("Shear image.jpg", img_shear)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif option ==8:
        #Reconocimiento facial
        faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(prueba)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceClassif.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            maxSize=(800,800))

        for (x,y,w,h) in faces:
            cv2.circle(img, (int(x + w/2), int(y + h/2)), int(h/2), (255, 0, 0), 2)


        # Obtener las dimensiones de la imagen
        height, width = img.shape[:2]

        # Generar la matriz de transformación shear
        matrix = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)

        # Aplicar el efecto shear
        img_shear = cv2.warpAffine(img, matrix, (width, height))

        #Rotar la imagen 90°
        imgR = cv2.rotate(img_shear, cv2.ROTATE_90_CLOCKWISE)

        #Reflejar la imagen
        flip0 = cv2.flip(imgR,0)

        #Reducir la resolución espacial a 72dpi
        im_ = Image.fromarray(flip0)

        im_.info['dpi'] = (72, 72)

        #Mostrar canal rojo
        imge = np.array(im_)

        #Separar los canales de color de la imagen original
        b, g, r = cv2.split(imge)

        imge[:,:,0]=0
        imge[:,:,1]=0
        
        #Guardar la imagen modificada en disco duro
        cv2.imwrite("canal_rojo.png", imge)

        #convertir la imagen a un array numpy
        im_array = np.array(imge)

        #aplicar una función de escalamiento para reducir el número de niveles de intensidad a 16
        im_array16 = (im_array // 16)*  16

        #convertir el array numpy de nuevo a una imagen PIL
        im_16 = Image.fromarray(im_array16)

        #guardar la imagen con 16 niveles de intensidad
        im_16.save("prueba_16.png")

        ##Imagen a 2 niveles de intesidad
        im2 = im_.convert('1')

        #guardar la imagen para convertir a 2 niveles de intensidad
        im2.save("prueba_2.png")

        # Mostrar imagen final
        cv2.imshow("canal_rojo.png", imge)
        cv2.waitKey(0)
        
    else:
        print("Opcion no valida.")

if __name__ == "__main__":
    menu()

