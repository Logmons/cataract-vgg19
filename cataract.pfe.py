import os  # Module pour interagir avec le système d'exploitation (gestion des fichiers, dossiers, etc.)
import numpy as np  # Bibliothèque pour l'algèbre linéaire et les calculs numériques
import pandas as pd  # Bibliothèque pour la manipulation et l'analyse des données
import cv2  # OpenCV pour le traitement des images
import random  # Module pour générer des nombres aléatoires
from tqdm import tqdm  # Permet d'afficher une barre de progression lors des boucles
import matplotlib.pyplot as plt  # Bibliothèque pour la visualisation des données
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Générateur d'images pour l'augmentation des données

# Chargement du fichier CSV contenant les données
df = pd.read_csv("./input/full_df.csv")

# Affichage des 3 premières lignes du DataFrame pour vérifier le contenu
print (df.head(3))
# Fonction pour vérifier si un texte contient le mot "jhcataract"
def has_cataract(text):
    if "cataract" in text:
        return 1
    else:
        return 0
    # Création d'une nouvelle colonne "left_cataract" indiquant la présence de cataracte à gauche
df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
# Création d'une nouvelle colonne "right_cataract" indiquant la présence de cataracte à droite
df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x))

left_cataract = df.loc[(df.C ==1) & (df.left_cataract == 1)]["Left-Fundus"].values
print (left_cataract[:15])

right_cataract = df.loc[(df.C ==1) & (df.right_cataract == 1)]["Right-Fundus"].values
print (right_cataract[:15])

print("Number of images in left cataract: {}".format(len(left_cataract)))
print("Number of images in right cataract: {}".format(len(right_cataract)))

left_normal = df.loc[(df.C ==0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250,random_state=42).values
right_normal = df.loc[(df.C ==0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250,random_state=42).values
right_normal[:15]

cataract = np.concatenate((left_cataract,right_cataract),axis=0)
normal = np.concatenate((left_normal,right_normal),axis=0)
# Affichage du nombre total d'images atteintes de cataracte et du nombre total d'images normales
print(len(cataract),len(normal))

from tensorflow.keras.preprocessing.image import load_img,img_to_array
dataset_dir = "./input/prep"
image_size=224
labels = []
dataset = []
# Fonction pour créer un dataset d'images avec leurs labels
def create_dataset(image_category,label):
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir,img)
        try:
            # Chargement de l'image en couleur avec OpenCV
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))
        except:
            continue # Si une erreur se produit (fichier non trouvé, format incorrect), on passe à l'image suivante
        
        # Ajout de l'image et du label sous forme de tableau NumPy dans la liste dataset
        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    return dataset
        
dataset = create_dataset(cataract,1)
print("Nombre d'images atteintes de cataracte :", len(dataset))
dataset = create_dataset(normal,0)
print("Nombre d'images normales :", len(dataset))
# Affichage de 10 images aléatoires du dataset
plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(dataset)))
    image = dataset[sample][0]
    category = dataset[sample][1]
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel(label)
    plt.tight_layout()    
# Création d'un tableau NumPy contenant toutes les images et les labels
x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
y = np.array([i[1] for i in dataset])
from sklearn.model_selection import train_test_split
# Séparation des données en ensembles d'entraînement et de test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Création du générateur d'entraînement avec augmentation des données
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Générateur pour les données de validation (sans augmentation, juste normalisation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Application du générateur sur les données numpy
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
val_generator = val_datagen.flow(x_test, y_test, batch_size=32)

from tensorflow.keras.applications.vgg19 import VGG19
# Chargement du modèle VGG19 pré-entraîné sans la couche de classification finale (include_top=False)
vgg = VGG19(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))
# Désactivation de l'entraînement de toutes les couches du modèle VGG19
for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras import Sequential  # Importation du modèle séquentiel de Keras
from tensorflow.keras.layers import Flatten, Dense  # Importation des couches nécessaires

from tensorflow.keras.layers import Dropout  # 👉 Mets ça en haut avant le modèle

# Création d'un modèle séquentiel
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))



# Affichage du résumé du modèle pour inspecter la structure
model.summary()

# Création d'un modèle nommé "Cataract_Model"
Model = Sequential(name="Cataract_Model")

# Compilation du modèle avec l'optimiseur "adam", la fonction de perte "binary_crossentropy" 
# pour une classification binaire, et la métrique "accuracy" pour évaluer les performances
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint  # Importation de la fonction de rappel pour enregistrer le modèle pendant l'entraînement

# Définition d'un callback pour enregistrer le modèle avec la meilleure validation
checkpoint = ModelCheckpoint("vgg19.keras", 
                             monitor="val_accuracy",  # Surveille la précision de validation
                             verbose=1,  # Affiche des informations pendant l'entraînement
                             save_best_only=True,  # Enregistre uniquement si la précision de validation est améliorée
                             save_weights_only=False,  # Enregistre le modèle complet, pas seulement les poids
                             save_freq='epoch')  # Sauvegarde à chaque époque


from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Entraînement du modèle avec les données d'entraînement, de validation et les callbacks
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    verbose=1,
    callbacks=[checkpoint, reduce_lr]
)

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(x_test, y_test)  # Calcule la perte et la précision sur l'ensemble de test
print("loss:", loss)  # Affiche la perte (loss)
print("Accuracy:", accuracy)  # Affiche la précision (accuracy)

# Importation des outils pour générer des métriques de classification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Prédiction des labels sur les données de test et conversion des probabilités en classes (0 ou 1)
y_pred = (model.predict(x_test) > 0.5).astype("int32")  # ✅ Si la probabilité est supérieure à 0.5, classe 1 (cataracte), sinon 0 (normal)

# Calcul de la précision en comparant les valeurs prédites avec les valeurs réelles
accuracy_score(y_test, y_pred)  # Calcul de la précision globale

# Importation de la fonction pour afficher la matrice de confusion
from mlxtend.plotting import plot_confusion_matrix

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
plot_confusion_matrix(conf_mat=cm, figsize=(8,7), class_names=["Normal", "Cataract"], show_normed=True);

# Importation de la bibliothèque matplotlib pour personnaliser l'affichage
import matplotlib.pyplot as plt

# Définition du style pour les graphiques
plt.style.use("ggplot")

# Création d'une figure pour visualiser les résultats (bien que la visualisation semble incomplète ici)
fig = plt.figure(figsize=(12,6))

# Ajuster la longueur des époques selon l'historique
epochs = range(1, len(history.history["accuracy"]) + 1)

# Tracer la courbe d'accuracy
plt.subplot(1,2,1)
plt.plot(epochs, history.history["accuracy"], "go-")
plt.plot(epochs, history.history["val_accuracy"], "ro-")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc="upper left")

# Tracer la courbe de loss
plt.subplot(1,2,2)
plt.plot(epochs, history.history["loss"], "go-")
plt.plot(epochs, history.history["val_loss"], "ro-")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")

# Affichage du graphique
plt.tight_layout()  # Assure que les sous-graphes ne se chevauchent pas
plt.show()
plt.figure(figsize=(12,7))
for i in range(15):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"
        
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"
        
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 


