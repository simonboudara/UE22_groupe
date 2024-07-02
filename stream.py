import streamlit as st
import pandas as pd
from tensorflow.keras import layers, models
import numpy as np 
import matplotlib.pyplot as plt
import random
from PIL import Image
import time
from io import BytesIO




st.sidebar.title('Sommaire')
pages = ['Présentation du projet', 'Exploration des données', 'Analyse des données','Comparaison fine-tuning et Transfert-Learning']

page = st.sidebar.radio("Aller vers la page", pages)

if page == pages[0] :
    st.write(''' 
## UE 22 - Machine Learning
#Contexte du projet
blablabla
         ''')
    


if page == pages[2] :
    st.write( ''' 
## Mise en place de notre modèle CNN et analyse ''')
    st.write('''
Ici je veux utiliser les commandes CNN donc surtout le truc que t'avais fait Grégor avec les filtre
Puis, je veux faire apparaître l'image qui est choisie (avec les commandes streamlit on peut zoomer dessus etc
Ensuite, j'essayerai de faire en animation les différentes étapes des filtres 
Sur le côté je vais mettre des barres avec le numéro du filtre (qui bouge avec une barre) et le numéro de l'étape (pareil)
Ce qu'il serait cool c'est qu'en bougeant sur les barres, on peut remonter à l'image qu'on veut
            ''')

    st.title("Architecture du Modèle de Convolution")
    
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

   
    df =model2.summary()
    st.write(df)
    model2.compile(optimizer='adam',loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
    

    frame_text = st.sidebar.empty()
    image_choisie = st.empty()
    image = st.empty()

    animation_speed = 0.1 

    conv1_model = tf.keras.Model(inputs=model2.inputs,outputs=model2.layers[0].output)
    conv2_model =tf.keras.Model(inputs=model2.inputs,outputs=model2.layers[2].output)
    conv3_model = tf.keras.Model(inputs=model2.inputs,outputs=model2.layers[4].output)
    conv4_model = tf.keras.Model(inputs=model2.inputs,outputs=model2.layers[6].output)
    conv5_model = tf.keras.Model(inputs=model2.inputs,outputs=model2.layers[8].output)
    conv6_model = tf.keras.Model(inputs=model2.inputs,outputs=model2.layers[10].output)

    classes = [os.path.join('/content/PBC_dataset_normal_DIB_224/PBC_dataset_normal_DIB_224',name) for name in os.listdir('/content/PBC_dataset_normal_DIB_224/PBC_dataset_normal_DIB_224')]
    random_classe = random.choice(classes)
    images = [os.path.join(random_classe, name) for name in os.listdir(random_classe)]
    random_image = random.choice(images)
    st.write('''"L'image initiale choisie par le programme est: 
             ''')
    image_choisie.image(random_image)
    image_c = tf.keras.preprocessing.image.load_img(random_image, target_size=(360, 363, 3))
    image_b=image_c

    image_b=tf.keras.preprocessing.image.img_to_array(image_b)
    image_b = tf.keras.applications.vgg16.preprocess_input(image_b)
    image_b = tf.expand_dims(image_b, axis=0)
    


    image_c = tf.keras.preprocessing.image.img_to_array(image_c)
    image_c = tf.keras.applications.vgg16.preprocess_input(image_c)
    image_c = tf.expand_dims(image_c, axis=0)
    image_b = image_c

    layer_number = st.sidebar.empty()
    filter_number = st.sidebar.empty()

    set_up = True
    
    while set_up == True :
        conv_output = conv1_model.predict(image_c)
        for j in range (0,32):
            st.write('''Couche 1 filtre, j 
                     ''')
            layer_number.write('Couche 1')
            filter_number.write(f'Couche 1 filtre {j}')
            filter_image = conv_output[0, :, :, j]
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
  

        conv_output2= conv2_model.predict(image_c)
  

        for j in range (0,64):

            st.write('''Couche 2 filtre, j 
                     ''')
            layer_number.write('Couche 2')
            filter_number.write(f'Couche 2 filtre {j}')
            filter_image = conv_output2[0, :, :, j]
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)


        conv_output3 = conv3_model.predict(image_c)

        for i in range (0,64):
            st.write('''Couche 3 filtre, i 
                     ''')
            layer_number.write('Couche 3')
            filter_number.write(f'Couche 3 filtre {i}')
            filter_image = conv_output2[0, :, :, i]
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
    
        conv_output4 = conv4_model.predict(image_c)
        for i in range (0,64):
            st.write('''Couche 4 filtre, i 
                     ''')
            layer_number.write('Couche 4')
            filter_number.write(f'Couche 4 filtre {i}')
            filter_image = conv_output2[0, :, :, i]
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
 

        conv_output5 = conv5_model.predict(image_c)
        for i in range (0,64):
            st.write('''Couche 5 filtre, i
                     ''')
            layer_number.write('Couche 5')
            filter_number.write(f'Couche 5 filtre {i}')
            filter_image = conv_output2[0, :, :, i]
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)
 

        conv_output6 = conv6_model.predict(image_c)
        for i in range (0,64):
            st.write('''Couche 6 filtre, i
                     ''')
            layer_number.write('Couche 6')
            filter_number.write(f'Couche 6 filtre {i}')
            filter_image = conv_output2[0, :, :, i]
            buf = BytesIO()
            plt.imsave(buf, filter_image, cmap='viridis')
            buf.seek(0)
            image.image(buf)


        time.sleep(animation_speed)
        
        set_up = False

    if st.button("Re-run") :
        set_up = True


    filter_index = st.sidebar.slider("Sélectionnez un filtre", 0, 31, 0)

    layer_index = st.sidebar.selectbox("Sélectionnez une couche", 6, 0)

    filter_image = conv_output[layer_index][0, :, :, filter_index]
    buf = BytesIO()
    plt.imsave(buf, filter_image, cmap='viridis')
    buf.seek(0)
    image.image(buf)






