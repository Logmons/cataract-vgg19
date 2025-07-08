from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Charger le modèle une seule fois
MODEL_PATH = "vgg19.keras"  # Assurez-vous que le chemin vers votre modèle est correct
model = load_model(MODEL_PATH)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Crée le dossier si nécessaire
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Fonction de prédiction avec dessin de cadre
def detect_cataract(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    image_copy = image.copy()  # Faire une copie pour dessiner sur l'image
    image_resized = cv2.resize(image, (224, 224))  # Redimensionner à la taille attendue par le modèle
    image_resized = img_to_array(image_resized) / 255.0  # Normalisation
    image_resized = np.expand_dims(image_resized, axis=0)  # Ajouter une dimension pour le batch

    # Faire la prédiction
    pred = model.predict(image_resized)
    
    # Vérifier si la cataracte est détectée ou non
    if pred > 0.5:
        result = "Attention: Cataracte détectée"
        result_class = "cataract"  # Classe CSS pour la cataracte (rouge)
    else:
        result = "Pas de cataracte"
        result_class = "normal"  # Classe CSS pour l'absence de cataracte (vert)

    # Sauvegarder l'image modifiée avec un cadre
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "result_" + os.path.basename(image_path))
    cv2.imwrite(output_path, image_copy)

    return result, result_class, output_path  # Retourner le résultat, la classe CSS et le chemin de l'image modifiée

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    result_class = None
    image_path = None
    if request.method == "POST":
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Appel du modèle de prédiction
            result, result_class, image_path = detect_cataract(filepath)

    return render_template("index.html", result=result, result_class=result_class, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
