"""import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Charger les données MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# Normaliser les valeurs des pixels pour les mettre dans la plage [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Définition du modèle
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Évaluation du modèle sur les données de test
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

"""

