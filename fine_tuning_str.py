import tensorflow as tf
import io 
import streamlit as st
from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, Input, Lambda
from tensorflow.keras.models import Model

def app():

    st.title("Le Fine-Tuning")
    st.write("""Le Fine-Tuning repose sur le même principe que le Transfer Learning. En effet on conserve encore une fois les poids de la 
            couche de convolution d'un modèle pré-entrainé, mais on dégèle les poids de la dernière couche de convolution pour lui permettre
            de mieux s'adapter aux caractéristiques de nos images.
            """)

    st.write("**Voici le code du Fine-Tuning**")
    st.write("On ne détaille pas ici ce code car il est très semblable à celui du Transfer Learning")

    fine_tuning_code1 = """
    base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, Input, Lambda
    from tensorflow.keras.models import Model

    top_net = [
        Dropout(0.3, seed=2022, name='dropout'),
        Flatten('channels_last', name='flatten'),
        Dense(512, activation='relu', name='fc1'),
        Dense(256, activation='relu', name='fc2'),
        Dense(8, activation='softmax', name='predictions')
    ]

    inputs = Input(shape=IMG_SHAPE, name='input')

    preprocess_input = Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x), name='preprocess')
    x = preprocess_input(inputs)


    for layer in base_model.layers[1:]:
        x = layer(x)

    for layer in top_net[:-1]:
        x = layer(x)
    outputs = top_net[-1](x)

    model_ft = Model(inputs, outputs, name='WBC')

    LAST_CONV_BLOCK_NAME = 'block5_conv1'
    for layer in model.layers:
        if layer.name == LAST_CONV_BLOCK_NAME:
            break
        layer.trainable = False

    base_learning_rate = 0.0001
    model.compile(optimizer=Adam(learning_rate=base_learning_rate),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    """
    st.code(fine_tuning_code1)

    st.write("Avec un model_ft.summary() on a : ")

    st.image('summary ft.png')

    st.write("**Résultat de l'entrainement**")

    st.image("entrainement vrai FT.png")

    st.write("**Test de l'algorithme**")

    model_tl_exploitable = tf.keras.models.load_model('model_tl_360_363')

    col1, col2 = st.columns(2)

    with col1 :
        st.write('Accuracy sur 1 essai')
        if st.button('1 essai', key='btn5'):
            DATASET = "snkd93bnjr-1\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB"
            train_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
            valid_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
            test_ds = valid_ds.take(2)
            valid_ds = valid_ds.skip(2)
            acc = model_tl_exploitable.evaluate(test_ds)[1]
            st.write("**L'accuracy sur 1 essai est de**", acc)

    with col2 :
        st.write('Accuracy sur 5 essais')
        if st.button('5 essais',key='btn6'):
            DATASET = "snkd93bnjr-1\PBC_dataset_normal_DIB\PBC_dataset_normal_DIB"
            train_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
            valid_ds = tf.keras.utils.image_dataset_from_directory(DATASET, validation_split=0.2, subset='training', image_size=(360,363), seed=42, batch_size=16)  # set params
            test_ds = valid_ds.take(2)
            valid_ds = valid_ds.skip(2)
            acc = 0
            for i in range(5):
                acc+= model_tl_exploitable.evaluate(test_ds)[1]
            st.write("**L'accuracy sur 5 essais est de**", acc/5)

