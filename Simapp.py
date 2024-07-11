import streamlit as st
import pandas as pd
 
def app():
    st.title("Exploration de données")

    st.header("Motivation")

    st.write("Avant d'implémenter un réseau de neurones qui permet de classifier les images de notre dataset, il peut être préférable de prendre en main notre jeu de données et en extraire les caractéristiques principales. C'est l'objectif de cette partie.")

    st.header("Création d'une première dataframe à partir de notre dataset")

    image_path = "Dataset1.png"

    st.image(image_path, caption="première dataset", use_column_width=True)

    st.write("La première chose intéressante à vérifier pourrait être que toute les images possédent la même dimension. En effet si ce n'est pas le cas, cela posera problème au niveau de la couche d'entrée du réseau de neurones à implémenter. ")

    image_path = "Image_verif_taille.png"

    st.image(image_path, caption=" ", use_column_width=True)

    st.write("Au vu du nombre d'image à notre disposition, on a choisi de supprimer les images qui n'ont pas la taille 363*360. ")

    image_path = "Image hauteur_bon.png"

    st.image(image_path, caption=" ", use_column_width=True)

    st.write("On se retrouve donc au final avec une dataframe avec des images de même taille et avec pour seule caractéristique le nom du globule")

    st.header("Création d'une dataframe avec de nouvelles observations")

    st.write("Le réseau de neurones que l'on va vous présenter dans la partie suivante va sélectionner les paramètres optimaux pour classifier les images. En effet dans notre cas il y a un pré-réseau (appelé réseau de neurones à convolusion) qui va permettre de sélectionner les meilleures paramètres pour que le réseau de classification puisse fonctionner de manière optimale. L'idée ici est de faire le travail du réseau de neurones à convolusion à la main. En effet, en regardant une à une les différentes images, nous allons créé des caractéristiques qui, selon nous,  permettent de bien classifier les globules." )

    st.subheader("Détermination des nouvelles caractéristiques pour la classification :")

    image_path1 = "BA_2862.jpg"
    image_path2 = "BNE_4555.jpg"
    image_path3 = "LY_25363.jpg"

    # Créer trois colonnes
    col1, col2, col3 = st.columns(3)

    # Afficher les images dans les colonnes respectives
    with col1:
        st.image(image_path1, caption="Basophil", use_column_width=True)
        
    with col2:
        st.image(image_path2, caption="Neutrophil", use_column_width=True)
        
    with col3:
        st.image(image_path3, caption="Lymphocyte", use_column_width=True)

    st.write("Choix des caractéristiques :")
    st.markdown("- Niveau de gris\n- Nombre de petits contours\n- Nombre de gros contours\n- Nombre de pixels supérieur à un certain seuil")

    st.write ("*Remarque : pour calculer chacune des ces caractéristiques, nous avons au préalable centré les images*")

    st.write ("En appliquant différents codes sur Google collab pour calculer ces nouvelles caractéristiques, on obtient la dataframe suivante : ")
    image_path = "Dataframe_finale.png"

    st.image(image_path, caption="Titre", use_column_width=True)


    st.write("On souhaite maintenant voir si les données que nous avons sélectionné à la main sont effectivement bonne pour classifier nos globules. Pour ce faire, on peut dans un premier temps avoir les histogrammes suivants : ")
    image_path1 = "Niveaugris.png"
    image_path2 = "Im_nombre de pixels.png"


    # Créer trois colonnes
    col1, col2 = st.columns(2)

    # Afficher les images dans les colonnes respectives
    with col1:
        st.image(image_path1, caption="niveu de gris", use_column_width=True)
        
    with col2:
        st.image(image_path2, caption="nombre de pixels", use_column_width=True)
        
    image_path1 = "Im_petits_contours.png"
    image_path2 = "Im_gros_contours.png"


    # Créer trois colonnes
    col1, col2 = st.columns(2)

    # Afficher les images dans les colonnes respectives
    with col1:
        st.image(image_path1, caption="petits contours", use_column_width=True)
        
    with col2:
        st.image(image_path2, caption="gros contours", use_column_width=True)

    st.subheader("Analyse des résultats")
    st.write("De manière générale, aucune des caractéristiques sélectionnées ne permet de bien classifier tous les globules dans toutes les catégories. Ainsi nous pouvons dire que sans traitement préalable des caractéristiques, il va être difficile pour des algorithmes de classification (comme les k plus proches voisins ou des arbres de décision) de bien classifier les donnéees.")

