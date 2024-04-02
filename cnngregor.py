## Organisation des data 
import os
from PIL import Image
import numpy as np
import re 

chemin = r"C:\Users\ronci\Documents\Ecole Mines\Cours\Info\PBC_dataset_normal_DIB_224\PBC_dataset_normal_DIB_224"
def load_images_and_labels(directory):
    image_data = []
    image_names = []
    image_labels =[]   
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpeg") or file.endswith(".jpg"):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    if img.size == (224, 224):
                        img_data = np.array(img)
                        image_data.append(img_data)
                        label = re.findall('[A-Z]+', file.split('_')[0])  # Récupère les lettres capitales avant '_'
                        image_labels.append(label[0])
                        image_names.append(file)
                    else : 
                        print("L'image n'est pas à la bonne dimension. Supression de ", file)
                except Exception as e:
                    print(f"Erreur lors du chargement de l'image {file}: {e}")
    image_data = np.array(image_data)   
    return image_data, image_names, image_labels


images, image_names, image_labels = load_images_and_labels(chemin)

## Réseau de neurones

# Création des jeux de données
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, image_labels, test_size=0.2, random_state=42)
print(X_train.shape)
# Faire une division par 255 pour avoir des valeurs entre 0 et 1
# Création de la structure du CNN 

import tensorflow as ts 
from tensorflow.keras import layers, models 

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(13, activation='softmax')
])

model.compile(optimizer='adam',loss= 'categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

test_loss , test_acc = model.evaluate(X_test, y_test)
print("Accuracy", test_acc)