# CIApp (Compteur intelligent automatique)

## Description

CIApp est un programme utilisant l'intelligence artificielle pour analyser et améliorer les mouvements humains en temps réel. Il exploite **Mediapipe Pose Estimation** pour détecter les points caractéristiques du corps humain et propose une visualisation intuitive grâce à un squelette dessiné dynamiquement et une jauge d'amplitude. L'application est conçue pour aider les utilisateurs à maintenir une posture correcte et à suivre leurs performances d'exercice avec des retours en temps réel.

![CIApp Logo V1(1)](https://github.com/user-attachments/assets/ecb095f9-7105-4ee0-bd13-95dcd7ddfe44)

Vous pouvez télécharger le fichier executable depuis : https://sourceforge.net/projects/ciapp/

J'ai mis en place un système de login grâce à firebase afin de pouvoir analyser des données comme le nombre d'utilisateur, ce qui oblige l'utilisateur à se connecter avant de pouvoir atteindre l'interface principale du logiciel.

![youtube-video-gif(10)](https://github.com/user-attachments/assets/8f2c640e-5a11-4a8d-b9ad-5bf02c1a67df)



### A propos de l'auteur

Je suis Mathis Sommacal, étudiant en Master 2 spécialisé en Intelligence Artificielle et Automatique Avancée pour l'Énergie. Passionné par les nouvelles technologies et le sport, j'ai allié ces deux centres d'intérêt pour créer le projet CIApp. Ce projet, réalisé en parallèle de mes études, m'a permis de m'entraîner et de développer mes compétences en vision par ordinateur, notamment dans l'utilisation d'OpenCV et de modèles de computer vision. Il reflète mon engagement à maîtriser ces outils tout en apportant des solutions pratiques pour l'analyse biomécanique et l'amélioration des performances sportives. Mon objectif est de poursuivre mes études en doctorat dans le domaine du computer vision afin de contribuer à l'avancement de ce domaine qui me passionne.

## Fonctionnalités

### 1. **Détection et Visualisation du Corps Humain**
- Utilisation du modèle Mediapipe Pose pour obtenir les coordonnées des points caractéristiques (articulations, segments corporels).
- Dessin des points détectés et tracé de lignes pour représenter le squelette humain.
- Visualisation améliorée avec des lignes dynamiques :
  - Les lignes changent de couleur pour indiquer si la posture est correcte ou non.

### 2. **Correction de Posture**
- Intégration de conditions dynamiques pour des exercices spécifiques comme le gainage ou les pompes.
- Retour en temps réel pour aider à ajuster la position du corps.

  ![youtube-video-gif](https://github.com/user-attachments/assets/d8c1717a-b1c5-4157-86b1-dd47a9524c11)


### 3. **Comptage des Répétitions**
- Calcul dynamique des angles entre les segments corporels via la fonction `calculate_angle` :
  ```python
  def calculate_angle(a, b, c):
      """Calculate the angle between three points."""
      a = np.array(a)
      b = np.array(b)
      c = np.array(c)

      radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
      angle = np.abs(radians*180.0/np.pi)

      if angle > 180.0:
          angle = 360 - angle

      return angle
  ```
- Les angles et les coordonnées X/Y sont analysés pour incrémenter le compteur de répétitions lorsque les conditions sont remplies.
- ![youtube-video-gif(1)](https://github.com/user-attachments/assets/cbe43d23-c327-4965-9600-9dfe25676cab)

### 4. **Personnalisation de la Caméra**
- Choix de la caméra via un slider (0 par défaut pour la webcam intégrée).
- Compatibilité avec des applications comme **iVCam** permettant d'utiliser un smartphone comme caméra distante.
  ![youtube-video-gif(8)](https://github.com/user-attachments/assets/a127ad88-ffaa-4b5b-af96-3644e0ee4f9d)


### 5. **Analyse Vidéo et Biomécanique**
- Téléchargement et traitement de vidéos enregistrées directement depuis un smartphone.
- Analyse temporelle des angles et coordonnées des points détectés pour une étude approfondie des mouvements.
- Utile pour l'analyse de mouvements dans divers sports (biomécanique, optimisation des performances).
![youtube-video-gif(5)](https://github.com/user-attachments/assets/d88841c0-c309-4219-82ee-43504097adc1)
## Installation
1. **Prérequis** :
   - Python 3.9 ou supérieur
   - Bibliothèques requises : Mediapipe, OpenCV, NumPy, Matplotlib, tkinter, customtkinter, pandas, requests, firebase-admin, openpyxl
   
2. **Installation des dépendances** :
   ```bash
   pip install mediapipe opencv-python numpy matplotlib tk customtkinter firebase-admin openpyxl
   ```

3. **Exécution de l'application** :
   ```bash
   python ciapp_beta.py
   ```

## Utilisation
1. **Mode Temps Réel** :
   - Lancez l'application.
   - Sélectionnez votre caméra via le slider.
   - Effectuez vos exercices devant la caméra et recevez un retour immédiat sur votre posture et vos répétitions.

2. **Mode Vidéo** :
   - Chargez une vidéo enregistrée depuis votre téléphone ou un autre appareil.
   - Analysez les mouvements, angles, et coordonnées.
   - Les résultats sont exportables vers Excel pour une étude plus poussée.
     ![youtube-video-gif(6)](https://github.com/user-attachments/assets/45e8f033-b312-4eb4-906e-7240a4ce3cc7)
## Avantages
- Amélioration de la posture et des performances sportives.
- Outil pratique pour les amateurs de fitness, entraîneurs et chercheurs en biomécanique.
- Compatible avec tous les sports grâce à ses fonctionnalités flexibles.

## Compatibilité
- Ordinateurs équipés de webcams ou de caméras distantes (via iVCam).
- Téléphones pour l'enregistrement de vidéos et une utilisation ulterieur via l'ordinateur pour une utilisation optimisée en salle de sport.

## Contributions
Les contributions sont les bienvenues ! Si vous souhaitez ajouter des fonctionnalités ou corriger des bugs, n’hésitez pas à soumettre une pull request ou à ouvrir une issue sur le dépôt GitHub.

## Licence
Ce programme est distribué gratuitement pour la communauté sous la licence BSD 3-Clause.

Vous êtes libre d'utiliser, modifier et redistribuer ce programme, tant que vous respectez les conditions suivantes :

    Mention des droits d'auteur : Vous devez conserver la notice de copyright présente dans le code source ou dans les documents associés.
    Limitation d'utilisation des noms : Le nom du détenteur des droits d'auteur (Mathis Sommacal) ou des contributeurs ne peut pas être utilisé pour promouvoir des produits dérivés sans autorisation écrite préalable.
    Exclusion de garantie : Ce programme est fourni "tel quel", sans garantie d'aucune sorte, implicite ou explicite.

Pour plus de détails, consultez le fichier LICENSE.

