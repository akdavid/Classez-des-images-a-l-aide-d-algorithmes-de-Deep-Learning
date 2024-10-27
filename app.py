import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Dossier des modèles sauvegardés et des classes de chiens
run_dir = './run_transfer_mobilnetv2'
figs_dir = './figs_transfer_mobilnetv2'

# Charger le modèle sauvegardé
model_path = f'{run_dir}/models/best_model.keras'
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# Dossier de destination contenant les images copiées
destination_dir = './selected_data/images/'

# Vérifier si le dossier de destination existe et récupérer les noms des classes
if os.path.exists(destination_dir):
    class_names = [breed_dir for breed_dir in os.listdir(destination_dir) if os.path.isdir(os.path.join(destination_dir, breed_dir))]
    class_names.sort()  # Trier les noms de classe par ordre alphabétique
else:
    class_names = []

# Paramètres de l'image
img_height = 224  # Adapter à la taille du modèle
img_width = 224


# Fonction pour prédire la race du chien à partir de l'image
def predict_breed(image):
    image = image.resize((img_width, img_height))
    img_array = ((np.array(image) / np.max(image))*255).astype(np.uint8)  # Normaliser les pixels
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter dimension batch
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_names[predicted_class]
    confidence = np.max(predictions)
    return predicted_label, confidence, predictions[0]


# Interface utilisateur avec Streamlit
st.title("Dog Breed Prediction")

st.write("""
         Téléchargez une image d'un chien et le modèle prédit la race !
         """)

# Télécharger une image
uploaded_file = st.file_uploader("Choisissez une image...",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Prédire la race du chien
    if st.button('Prédire la race'):
        st.write("Prédiction en cours...")
        predicted_label, confidence, probabilities = predict_breed(image)

        st.write(f"**Race prédite : {predicted_label}**")
        st.write(f"**Confiance : {confidence * 100:.2f}%**")

        # Afficher les probabilités pour chaque race
        st.write("### Probabilités pour chaque race :")
        fig, ax = plt.subplots()
        ax.barh(class_names, probabilities)
        ax.set_xlabel('Probabilité')
        ax.set_ylabel('Races de chiens')
        ax.set_xlim([0, 1])
        st.pyplot(fig)
