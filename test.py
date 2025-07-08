import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import argparse

# Charger le modèle une seule fois pour éviter de le recharger à chaque test
MODEL_PATH = "vgg19.keras"
model = load_model(MODEL_PATH)

# Fonction pour tester un modèle avec une image
def test_model(image_path, image_size=224):
    # Charger l'image et la redimensionner
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = img_to_array(image) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch

    # Faire une prédiction
    pred = model.predict(image)
    
    # Retourner le résultat de la classification
    return "Cataract" if pred > 0.5 else "Normal"

# Utilisation via terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester un modèle CNN sur une image.")
    parser.add_argument("image_path", type=str, help="Chemin vers l'image à tester")

    args = parser.parse_args()

    result = test_model(args.image_path)
    print(f"Prédiction pour {args.image_path}: {result}")
