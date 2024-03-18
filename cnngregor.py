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
import sklearn as sk

X_train, X_test, y_train, y_test = sk.train_test_split(images, image_labels, test_size=0.2, random_state=42)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Création de la structure du CNN 
