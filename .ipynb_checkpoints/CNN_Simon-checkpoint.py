import tensorflow as tf
from tensorflow.keras import layers, models

# Définir le modèle
def create_model():
    model = models.Sequential([
        # Couche d'entrée avec 64 neurones et fonction d'activation ReLU
        layers.Dense(64, activation='relu', input_shape=(784,)),
        # Couche cachée avec 64 neurones et fonction d'activation ReLU
        layers.Dense(64, activation='relu'),
        # Couche de sortie avec 10 neurones (classes) et fonction d'activation softmax pour la classification
        layers.Dense(10, activation='softmax')
    ])
    # Compiler le modèle avec une fonction de perte, un optimiseur et des métriques
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Charger les données (exemple avec le dataset MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train, x_test = x_train / 255.0, x_test / 255.0

# Créer une instance du modèle
model = create_model()

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5)

# Évaluer le modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)





