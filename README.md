## Fake News Detection – Multimodal

Ce projet propose un système de détection de fake news basé sur une approche multimodale combinant analyse de texte et d’images, avec enrichissement optionnel par Retrieval-Augmented Generation (RAG).

### Description

Le système vise à classifier un contenu comme fiable ou faux en exploitant simultanément plusieurs sources d’information. Il combine un encodeur textuel (DistilBERT), un extracteur visuel (ResNet-50) et une fusion des représentations pour la prédiction finale.

### Architecture

* **Backend** : API FastAPI exposant un endpoint `/predict` pour l’inférence multimodale
* **Frontend** : Interface Streamlit pour tester le modèle et visualiser les résultats
* **Pipeline** : prétraitement, encodage texte/image, fusion, classification, enrichissement RAG optionnel

### Données

Le modèle est entraîné et évalué sur le dataset multimodal **GossipCop**, après nettoyage et harmonisation des données texte-image.

### Fonctionnalités

* Classification fake news (texte + image)
* Traduction et normalisation linguistique
* Enrichissement via RAG
* Interface interactive Streamlit
* API de prédiction FastAPI
