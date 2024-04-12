import os
import pandas as pd
from PIL import Image

def create_csv_from_image_folder(input_folder, output_csv):
    """
    Crée un fichier CSV à partir d'un dossier contenant des images en utilisant Pandas.
    """
    image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    new_size=(360,363)
    for image_path in image_files :
        image = Image.open(image_path)
        if image.size != new_size :
            image_files.remove(image_path)

    df = pd.DataFrame({'image_path': image_files})
    df.to_csv(output_csv, index=False)

# Exemple d'utilisation
input_folder = r"C:\Users\ronci\Downloads\archive\snkd93bnjr-1\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB\Globule blanc\neutrophil"  # Chemin du dossier contenant les images
output_csv = r"C:\Users\ronci\Downloads\archive\snkd93bnjr-1\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB\Globule blanc CSV\neutrophil.csv"  # Nom du fichier CSV de sortie
create_csv_from_image_folder(input_folder, output_csv)


