import streamlit as st
import transfer_learning_str
import fine_tuning_str
import enregistrement_fichier_str
import Simapp
import conclusion
import notreCNN_str
import introduction_str
import théorie_cnn

# Dictionnaire des pages pour la navigation
pages = {"Présentation du problème" : introduction_str,
    "Analyse des data" : Simapp,
    "Théorie des réseaux" : théorie_cnn,
    "Notre CNN" : notreCNN_str,
    "Transfer Learning": transfer_learning_str,
    "Fine Tuning": fine_tuning_str,
    "Sauvegarde des modèles": enregistrement_fichier_str,
    "Conclusion" : conclusion
}

# Menu déroulant pour sélectionner la page
st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Aller à la page", list(pages.keys()))

# Affichage de la page sélectionnée
page = pages[selection]
page.app()