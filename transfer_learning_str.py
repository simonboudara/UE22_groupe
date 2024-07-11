import transfer_learning_str as st
import tensorflow as tf
import io 
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, Input, Lambda
from tensorflow.keras.models import Model


def app():

    # Titre de la page
    st.title("Transfer Learning")

    # Introduction
    st.write("""
    Après avoir travaillé sur notre propre réseau de neurones, voyons deux méthodes utilisant des réseaux pré-entrainés 
    afin de tirer parti d'une puissance de calcul plus importante tout en restant sur nos modestes machines. Sur cette page nous nous concentrerons
    la méthode du **Transfer Learning**
    """)

    # Section Transfert Learning
    st.header("Transfer Learning")
    st.write("""
    Le Transert Learning est une méthode consistant à reprendre la partie convolutive d'un réseau pré-entrainé (ici VGG16)
    et d'y "coller" un top-net non-entrainé et personnalisé. Le partie convolutive est donc ici entrainé sur des millions 
    d'images qui n'ont rien à voir avec les globules blancs, puis ses poids sont gelés et le top-net s'entraine sur nos 
    images. 
    """)
    st.image('Transfer_learning.svg')


    st.write("**Voici notre code de Transfert Learning à partir de VGG16**")

    transfert_learning_code1 = '''
    import tensorflow as tf
    from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, Input, Lambda
    from tensorflow.keras.models import Model

    base_model = tf.keras.applications.VGG16(input_shape=(360,363,3), include_top=False, weights='imagenet')
    base_model.trainable = False

    top_net = [
        Dropout(0.3, seed=2022, name='dropout'),
        Flatten('channels_last', name='flatten'),
        Dense(512, activation='relu', name='fc1'),
        Dense(256, activation='relu', name='fc2'),
        Dense(8, activation='softmax', name='predictions')
    ]

    # Créer une première couche d'INPUT -> entrée de la prochaine couche du CNN f(g(x))
    inputs = Input(shape=IMG_SHAPE, name='input')

    # On embarque la couche de preprocessing dans une fonction Lambda car sinon on arrive pas à sauvegarder le modèle
    preprocess_input = Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x), name='preprocess')
    x = preprocess_input(inputs)

    # x permet de créer un graphe de liasion entres les différentes couches 
    for layer in base_model.layers[1:]:
        x = layer(x)

    # On ajoute notre réseau de classification à 8 classes
    for layer in top_net[:-1]:
        x = layer(x)
    outputs = top_net[-1](x)

    model_tl = Model(inputs, outputs, name='WBC')
    '''
    st.code(transfert_learning_code1,language='python')
    
    st.write("Avec un model_tl.summary() on a : ")
    
    st.image('summary tl.png')

    st.write("Après avoir il faut compiler notre modèle en lui attribuant une fonction coût et un optimiseur : ")

    transfert_learning_code2 ="""
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import SparseCategoricalCrossentropy

    base_learning_rate = 0.0001
    model_tl.compile(optimizer=Adam(learning_rate=base_learning_rate),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    """

    st.code(transfert_learning_code2, language ='python')

    st.write("Puis enfin on entraine notre réseau, à vrai dire on n'entraine que notre top-net car on a gelé les paramètres de la partie convolutive")

    transfert_learning_code3 = """
    history = model_tl.fit(train_ds, epochs=15, validation_data=valid_ds)
    """

    st.code(transfert_learning_code3)

    st.write("**Résultat de l'entrainement**")

    st.image('entrainement TL_learning.png')

    st.write("**Enregistrement du modèle**")

    st.write("Une fois notre modèle entrainé sur Colab, on enregistre les poids et l'architecture pour pouvoir l'exécuter sur nos machines. Pour plus de détails, cf la page : Enregesitrement des modèles")

    st.write("**Test de l'algorithme**")

    model_tl_exploitable = tf.keras.models.load_model('model_tl_360_363')

    col1, col2 = st.columns(2)

    with col1 :
        st.write('**Accuracy sur 1 essai**')
        if st.button('1 essai',key='btn2'):
                IMG_SHAPE = (360,363,3)
                DATASET = "snkd93bnjr-1\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB"
                train_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
                valid_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
                test_ds = valid_ds.take(2)
                valid_ds = valid_ds.skip(2)
                acc = model_tl_exploitable.evaluate(test_ds)[1]
                st.write("**L'accuracy sur 1 essai est de**", acc)

    with col2 :
        st.write('**Accuracy sur 5 essais**')
        if st.button('5 essais',key='btn3'):
            acc = 0
            IMG_SHAPE = (360,363,3)
            DATASET = "snkd93bnjr-1\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB"
            train_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
            valid_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
            test_ds = valid_ds.take(2)
            valid_ds = valid_ds.skip(2)
            for i in range(5):
                acc+= model_tl_exploitable.evaluate(test_ds)[1]
            st.write("**L'accuracy sur 1 essai est de**", acc/5)