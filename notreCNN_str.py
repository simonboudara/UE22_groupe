import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np 
import matplotlib.pyplot as plt
import random
from PIL import Image
import time
import io
from io import BytesIO
import os

def app():
    st.write( ''' 
    ## Mise en place de notre modèle CNN et analyse ''')
    st.write('''
    Les Convolutional Neural Networks (CNN) sont une classe de réseaux de neurones particulièrement adaptés au traitement et à l'analyse des données visuelles.''')

    st.title("Architecture du Modèle de Convolution du Net")

    model_code1 = ''' from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense

    model1 = Sequential([
    InputLayer(input_shape=(28, 28, 3), name='input'), #input d'entrée rgb, juste couche crée vide


    Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='sigmoid', name='conv1'), #16 filtres différents dans la couche, filtre = matrice qui passe sur ton image en faisant produit matriciel. Padding comment je gère les bords
    MaxPool2D(strides=2, name='max1'), #diminution dimension : padd il fait une selection pour diminuer dimension en prenant ici spécifiquement pixel max intensité, c'est les dimensions images qui prennent
    Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='sigmoid', name='conv2'), #même chose que conv2D_1
    MaxPool2D(strides=2, name='max2'),
    Flatten(name='flat'),

    Dense(120, activation='sigmoid', name='dense1'),
    Dense(84, activation='sigmoid', name='dense2'),
    Dense(8, activation='softmax', name='classifier')
    ], name='LeNet')
    '''
    st.code(model_code1, language='python')
    st.write(''' Ce modèle de convolution du net renvoie une accuracy de 0.74 en moyenne, avec des images de taille (28,28). ''')

    model_code2 = ''' from tensorflow.keras import layers, models

    model2= models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='softmax')
    ])'''

    st.code( model_code2, language = 'python')
    st.write(''' Ce modèle de convolution renvoie une accuracy de 0.94 en moyenne, avec des images de taille (28,28). ''')

    model_code = '''model2= models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(360, 363, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,(2,2),activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(8, activation='softmax')
    ])'''

    st.code(model_code, language='python')
    st.write(''' Ce modèle de convolution renvoie une accuracy de 0.98 en moyenne, avec des images de taille (360,363). ''')


    model2= models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(360, 363, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,(2,2),activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(8, activation='softmax')
    ])

    st.write( ''' On peut montrer le sommaire de ce programme, qui témoigne des différents paramètres entraînés ou non, et d'autres spécificités.''')

    def get_model_summary(model):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()
        stream.close()
        return summary_str

    if st.button("Afficher le sommaire du modèle"):
        summary = get_model_summary(model2)
        st.text(summary)

    st.title("Analayse des différentes étapes du modèle de convolution")
    st.write('''
    ### Choix d'une image ''')
    st.write(''' 
    On choisit aléatoirement une image dans la dataset''')

    image_random = st.empty()
    st.image('MMY_270358.jpg')

    frame_text = st.sidebar.empty()
    st.write('''
    ### Etude plus précise de chaque étape ''')
    st.write(''' On peut également choisir précisément l'image traitée par la couche et le filtre voulus en déplaçant les barres latérales.''')
    image_choisie = st.empty()
    placeholder = st.empty()


    layer_number = st.sidebar.empty()
    filter_number = st.sidebar.empty()

    layer_index = st.sidebar.slider("Sélectionnez une couche", 1, 6, 1)
    filter_index = st.sidebar.slider("Sélectionnez un filtre", 0, 63, 0)
    image_path = rf"C:\Users\ronci\mines-paris-info\ProjetInfoS2_grouoe\activations_images3\Couche{layer_index}_filtre{filter_index}.png"


    filter_image = Image.open(image_path)
    buf = BytesIO()
    plt.imsave(buf, filter_image, cmap='viridis')
    buf.seek(0)
    image_choisie.image(buf)
    with placeholder.container():    
        st.write(f'Couche {layer_index} filtre {filter_index}')
    st.write('''
    ### Visualisation détaillée ''')
    st.write(''' 
    On peut afficher les différentes images renvoyées à chaque étape du réseau de neurones.''')

    image = st.empty()
    placeholder2 = st.empty()

    if st.button('Run'):

        if st.button("Stop"):
                st.stop()

        for j in range (0,32):
        
            image_path = rf"C:\Users\ronci\mines-paris-info\ProjetInfoS2_grouoe\activations_images3\Couche1_filtre{j}.png"
            filter_image = Image.open(image_path)
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
            with placeholder2.container():
                st.write(f'Couche 1 filtre {j}')
            time.sleep(1)


        for j in range (0,64):

            with placeholder2.container():
                st.write(f'Couche 2 filtre {j}')
            layer_number.write('Couche 2')
            filter_number.write(f'Couche 2 filtre {j}')
            image_path = rf"C:\Users\ronci\mines-paris-info\ProjetInfoS2_grouoe\activations_images3\Couche2_filtre{j}.png"
            filter_image = Image.open(image_path)
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
            time.sleep(1)

        for i in range (0,64):
            with placeholder2.container():    
                st.write(f'Couche 3 filtre {i}')
            layer_number.write('Couche 3')
            filter_number.write(f'Couche 3 filtre {i}')
            image_path = rf"C:\Users\ronci\mines-paris-info\ProjetInfoS2_grouoe\activations_images3\Couche3_filtre{i}.png"
            filter_image = Image.open(image_path)
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
            time.sleep(1)

        for i in range (0,64):
            with placeholder2.container():    
                st.write(f'Couche 4 filtre {i}')
            layer_number.write('Couche 4')
            filter_number.write(f'Couche 4 filtre {i}')
            image_path = rf"C:\Users\ronci\mines-paris-info\ProjetInfoS2_grouoe\activations_images3\Couche4_filtre{i}.png"
            filter_image = Image.open(image_path)
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
            time.sleep(1)

        for i in range (0,64):
            with placeholder2.container():   
                st.write(f'Couche 5 filtre {i}')
            layer_number.write('Couche 5')
            filter_number.write(f'Couche 5 filtre {i}')
            image_path = rf"C:\Users\ronci\mines-paris-info\ProjetInfoS2_grouoe\activations_images3\Couche5_filtre{i}.png"
            filter_image = Image.open(image_path)
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
            time.sleep(1)


        for i in range (0,64):
            with placeholder2.container():    
                st.write(f'Couche 6 filtre {i}')
            layer_number.write('Couche 6')
            filter_number.write(f'Couche 6 filtre {i}')
            image_path = rf"C:\Users\ronci\mines-paris-info\ProjetInfoS2_grouoe\activations_images3\Couche6_filtre{i}.png"
            filter_image = Image.open(image_path)
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
            time.sleep(1)


    st.title('Utilisation de Grad-Cam')
    st.write(''' Les réseaux de neurones ont pour but de traiter des données pour extraire des représentations utile à l’atteinte d’un objectif. C'est pour cela que nous 
    avons cherché à utiliser Grad-Cam. En effet, Grad-Cam est une méthode de visualisation des parties d'une image donnée qui ont permis 
    à un convnet de décider sur la classification finale. Il permet de comprendre ce que les CNN regardent réellement.''')
    st.write(''' 
    ### Carte thermique ''')
    st.write(''' La carte d’activation de classe pondérée par le gradient Grad-CAM produit une carte thermique qui met en évidence les régions importantes d’une image 
    en utilisant les gradients de la cible (ici le globule et ses spécificités) de la couche convolutive finale. En simple, elle indique
    dans quelle mesure une partie correspond à un lymphocyte, ou autre.''')

    image = st.empty()

    st.image('grad_cam2.jpg')


    st.write(''' 
    ### Superposition de l'image et du gradient ''')
    st.write(''' On peut ensuite superposer ce gradient thermique avec l'image considérée. Ainsi, on observe concrètement si les parties 
    considérées par le modèle pour trancher sont pertinentes. Cette méthode peut donc permettre de trouver le problème, si jamais.''')
    image2 = st.empty()

    st.image('superimposed_cam.png')











