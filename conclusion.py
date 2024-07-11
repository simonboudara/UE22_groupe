import streamlit as st
import pandas as pd

def app():
    st.title("Conclusion")

    st.header("ÉTAPES DU PROJET")
    # Fonction pour afficher un bloc de texte avec une classe CSS
    def display_text_block(text):
        st.markdown(f"<div class='text-block'>{text}</div>", unsafe_allow_html=True)

    st.markdown("""
        <style>
            .text-block {
                background-color: #FFEBCD; /* Orange clair */
                border: 2px solid #FFA500; /* Bordure orange */
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                text-align: center; /* Centrer le texte */
                font-weight: bold; /* Texte en gras */
            }
            .arrow {
                font-size: 36px;
                text-align: center;
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Contenu des blocs de texte
    text1 = "Exploration de données"
    text2 = "Implémentation de notre réseau de neurones et résultats"
    text3 = "Comparaison avec fine tuning et transfert learning"
    text4 = "Conclusion générale"

    # Afficher les blocs de texte et les flèches
    display_text_block(text1)
    st.markdown("<div class='arrow'>&#8595;</div>", unsafe_allow_html=True)  # Flèche vers le bas
    display_text_block(text2)
    st.markdown("<div class='arrow'>&#8595;</div>", unsafe_allow_html=True)  # Flèche vers le bas
    display_text_block(text3)
    st.markdown("<div class='arrow'>&#8595;</div>", unsafe_allow_html=True)  # Flèche vers le bas
    display_text_block(text4)