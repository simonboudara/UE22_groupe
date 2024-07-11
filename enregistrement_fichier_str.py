import streamlit as st

def app():
    st.title("Enregistrement des modèles")

    st.header("Contexte")

    st.write(""""
    Avec nos modestes ordinateurs sans carte graphique, l'entrainement d'un réseau de neurones à convolution est impossible. C'est pourquoi nous utilisons
            la puissance de calcul fournie gratuitement sur Google Colab pour entrainer nos modèles. Toutefois une fois les modèles entrainés,
            nous souhaiterions les enregistrer pour pouvoir les utiliser sur nos machines.
            """)

    st.header("Enregistrement du modèle")

    st.write("""TensorFlow permet de facilement enregistrer des modèles (architecture entière ou poids seulement) et les charger sur un autre
            environnement.
            """)

    st.write("**Enregistrement de l'architecture**")

    st.code("""
    model.save('nom_choisi_pour_enregistrement')
    """, language='python')

    st.write("**Enregistrement des poids**")

    st.code("""
    model.save_weights('nom_choisi_pour_enregistrer')
    """)

    st.write("**Chargement de l'architecture**")

    st.code("""
    model.load('nom_du_fichier_architecture')
    """)

    st.write("**Chargement des poids**")

    st.code("""
    model.load_weights('nom_du_fichier_poids')
    """)

    st.header("Difficulté rencontrée")

    st.write("Malheureusement l'enregistrement d'architecture entière à partir de Google Colab ne fonctionne pas et fait appraitre ce message d'erreur lorsque l'on souhaite le recharger sur nos machines")

    st.image('bug enre.jpg')

    st.write("Pour résoudre ce problème on enregistre les poids sur Google Colab, puis on enregistre l'architecture global directement sur notre machine")