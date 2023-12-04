import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

#Aqui se montó Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Aqui se definió el directorio que contiene las imágenes originales
directorio_originales = '/content/drive/MyDrive/Datos_red_prototipica/Imagenes_clasificadas'

#Aqui se definió el directorio donde se guardarán las imágenes rotadas
directorio_finales = '/content/drive/MyDrive/Datos_red_prototipica/Imagenes_rotadas'

#Aqui se obtuvo la lista de carpetas (géneros) en el directorio de imágenes originales
generos = os.listdir(directorio_originales)

#Aqui se inicializó una lista para almacenar los nombres de las imágenes rotadas
nombres_imagenes_rotadas = []

#Aquí se iteró a través de las carpetas (géneros)
for genero in generos:
    genero_path = os.path.join(directorio_originales, genero)

    #Aqui se creó una carpeta correspondiente en el directorio de imágenes finales
    carpeta_genero_finales = os.path.join(directorio_finales, genero)
    
    #Aqui se obtuvo la lista de imágenes en la carpeta del género
    archivos = os.listdir(genero_path)

    #Aqui se iteró a través de las imágenes en la carpeta del género
    for archivo in archivos:
        ruta_completa_original = os.path.join(genero_path, archivo)

        #Aquí se cargó la imagen original usando OpenCV
        imagen = cv2.imread(ruta_completa_original)

        if imagen is not None:
            #Aquí se inicializó un contador para los nombres de las imágenes rotadas
            contador = 2

            #Aquí se inicializó una lista para almacenar los nombres de las imágenes rotadas
            nombres_imagenes_rotadas_genero = []

            for numero_rotacion in [5, 10, 15]:
                #Aqui se rotaron las imagenes originales en los ángulos correspondientes mencionados anteriormente
                matriz_rotacion = cv2.getRotationMatrix2D((imagen.shape[1] / 2, imagen.shape[0] / 2), numero_rotacion, 1)
                imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (imagen.shape[1], imagen.shape[0]))

                #Aqui se generó el nombre de la imagen rotada con el formato diseñado
                nombre_imagen_rotada = f'{archivo.split(".")[0]}_{contador}_{numero_rotacion}deg.png'

                #Aquí se guardaron las imagenes rotadas en el directorio de imágenes finales 
                ruta_completa_rotada = os.path.join(carpeta_genero_finales, nombre_imagen_rotada)
                cv2.imwrite(ruta_completa_rotada, imagen_rotada)

                #Aqui se agregó el nombre a la lista del género
                nombres_imagenes_rotadas_genero.append(nombre_imagen_rotada)

                #Aqui se definió que el contador se Incrementar el contador
                contador += 1

            #Aqui se agregaron los nombres al registro general
            nombres_imagenes_rotadas.extend(nombres_imagenes_rotadas_genero)

#Aqui se imprimieron todos los nombres de las imágenes rotadas
for nombre in nombres_imagenes_rotadas:
    print(nombre)

directorio_rotadas = '/content/drive/MyDrive/Datos_red_prototipica/Imagenes_rotadas'

#Aqui se obtuvo la lista de carpetas (géneros) en el directorio de imágenes rotadas
generos = os.listdir(directorio_rotadas)

#Aqui se inicializó una lista para almacenar las imágenes de soporte
imagenes_soporte = []

#Aquí se inicializó un contador para el número total de imágenes rotadas
num_imagenes_rotadas = 0

#Aquí se iteró a través de las carpetas (géneros)
for genero in generos:
    genero_path = os.path.join(directorio_rotadas, genero)

    #Aquí se obtuvo la lista de archivos (imágenes) en la carpeta del género
    archivos = os.listdir(genero_path)

    #Aqui se iteró a través de los archivos (imágenes) en la carpeta del género
    for archivo in archivos:
        ruta_completa = os.path.join(genero_path, archivo)
        imagen = cv2.imread(ruta_completa)

        if imagen is not None:
            #Aquí se especificó que el contador de imágenes rotadas se incremente en 1
            num_imagenes_rotadas += 1

            #Aquí se redimensionaron las imagenes a 250x500 píxeles
            imagen = cv2.resize(imagen, (250, 500))
            #Aquí se normalizó la imagen dividiendo por 255 para obtener valores en el rango [0, 1]
            imagen = imagen / 255.0
            imagenes_soporte.append(imagen)

#Aquí se convirtió la lista de imágenes de soporte a un arreglo NumPy
imagenes_soporte = np.array(imagenes_soporte)

#Aqui se mostró la cantidad de imágenes rotadas
print("Número total de imágenes rotadas:", num_imagenes_rotadas)
print("Forma del conjunto de soporte:", imagenes_soporte.shape)

support_set = imagenes_soporte

#Aqui se calcularon las representaciones prototípicas para cada clase en el conjunto de soporte
unique_labels = np.arange(len(generos))  
prototypes = {}

for label in unique_labels:
    label_indices = np.where(unique_labels == label)[0]
    label_samples = support_set[label_indices]
    prototype = tf.reduce_mean(label_samples, axis=0)
    prototypes[label] = prototype

#Aqui se convirtieron las representaciones prototípicas en una matriz
prototype_matrix = tf.stack(list(prototypes.values()))

#Aqui se imprimió la forma de la matriz de prototipos
print("Forma de prototype_matrix:", prototype_matrix.shape)

#Aqui se dió formato a las imágenes de soporte
support_set = imagenes_soporte

#Aqui se generaron etiquetas de soporte para que coincidan con la cantidad de imágenes de soporte
support_labels = np.repeat(unique_labels, num_imagenes_rotadas // len(unique_labels))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(500, 250, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(50, activation='softmax')  # 50 clases (géneros)
])

#Aquí se compiló el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Aquí se entrenó el modelo utilizando el conjunto de soporte y las etiquetas de soporte
model.fit(support_set, support_labels, epochs=30, batch_size=32)

import matplotlib.pyplot as plt

#Aqui se definieron los datos de precisión
precision = [0.0225, 0.0213, 0.0250, 0.0275, 0.0312, 0.0225, 0.0275, 0.0200, 0.0162, 0.0125]

#Aqui se definió el número de épocas
epocas = range(1, len(precision) + 1)

#Aqui se creó el gráfico
plt.figure(figsize=(10, 6))
plt.plot(epocas, precision, marker='o', linestyle='-')
plt.title('Precisión a lo largo de las épocas')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.grid(True)

#Aqui se guardó la gráfica 
ruta_guardado = '/content/drive/MyDrive/Datos_red_prototipica/Grafica/precision_plot.png'
plt.savefig(ruta_guardado)

#Aqui se mostró la gráfica en pantalla 
plt.show()