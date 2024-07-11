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
    st.title("UE22 - Classification des globules blancs")
    st.write(''' 
        L'intelligence artificielle (IA) est un secteur en plein essor avec un avenir prometteur. Ses nombreuses applications contribuent à améliorer la médecine, 
        notamment à travers des opérations assistées, le suivi à distance des patients, des prothèses intelligentes, et des traitements personnalisés grâce au big data.
                
        ## Les globules blancs ''')
    st.image("illustration globblanc.png")
    st.write(''' Les globules blancs sont des cellules produites dans la moelle osseuse et présentes, entre autres, dans le sang les organes lymphoïdes et 
        de nombreux tissus conjonctifs de l'organisme. Il en existe plusieurs types. Chaque type joue un rôle important au sein du système immunitaire 
        en participant à la protection contre les agressions d'organismes extérieurs. Les types de globules blancs aident à diagnostiquer des troubles du sang et des pathologies.
        Pour cela, il est nécessaire de pouvoir quantifier le nombre 
        de globules blancs et leur type.''')
