import os
from PIL import Image

def preprocess_images_in_place(directory, target_size=(360, 363)):
    # Parcourir toutes les images dans le répertoire spécifié
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            
            # Redimensionner l'image
            img_resized = img.resize(target_size)
            
            # Sauvegarder l'image redimensionnée en écrasant l'originale
            img_resized.save(img_path)

            print(f"Image {filename} redimensionnée et sauvegardée.")

# Répertoire contenant les images
directory = r"C:\Users\chata\OneDrive\Documents\activation_image_info\activations_images2"
directory2 = r"C:\Users\chata\OneDrive\Documents\activation_image_info\activations_images2\activations_images3"

# Exécution de la fonction de prétraitement
#preprocess_images_in_place(directory)

#preprocess_images_in_place(directory2)




img_path = r"C:\Users\chata\Downloads\Blausen_0909_WhiteBloodCells.png"
img = Image.open(img_path)
            
            # Redimensionner l'image
img_resized = img.resize((360,363))
            
            # Sauvegarder l'image redimensionnée en écrasant l'originale
img_resized.save(img_path)

print(f"Image {'Blausen_0909_WhiteBloodCells.png'} redimensionnée et sauvegardée.")