
# Projet de classification de races de chiens

## Description du Projet

Vous êtes bénévole pour l'association de protection des animaux de votre quartier, Le Refuge. C'est d'ailleurs ainsi que vous avez trouvé votre compagnon idéal, Snooky. Vous vous demandez donc ce que vous pouvez faire en retour pour aider l'association.

Vous apprenez, en discutant avec un bénévole, que leur base de données de pensionnaires commence à s'agrandir et qu'ils n'ont pas toujours le temps de référencer les images des animaux qu'ils ont accumulées depuis plusieurs années.
Ils aimeraient donc obtenir un algorithme capable de classer les images en fonction de la race du chien présent sur l'image.

### Données

Les bénévoles de l'association n'ont pas eu le temps de réunir les différentes images des pensionnaires dispersées sur leurs disques durs. Pas de problème, vous entraînerez votre algorithme en utilisant le Stanford Dogs Dataset.

### Mission

L'association vous demande de réaliser un algorithme de détection de la race du chien sur une photo, afin d'accélérer leur travail d’indexation. 
Ce projet inclut plusieurs étapes allant de la préparation des données, à la conception de modèles CNN et au transfer learning.

## Installation

Pour installer les dépendances :
```
pip install -r requirements.txt
```

## Utilisation

Ce projet utilise Streamlit pour déployer une application de prédiction de races de chiens. Pour lancer l'application, exécutez :
```
streamlit run app.py
```
ou
```
streamlit run app_v2.py
```

La première version comporte 5 classes de chiens tandis que la v2 comporte 10 classes.

## Auteur

Anthony David
