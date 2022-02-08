# Neural_NetVox

### Contributeurs:

![contrib](https://user-images.githubusercontent.com/77715595/148669448-e9e532a4-9f37-4b19-af48-eba3a77c2a17.jpg)


Le projet consiste à extraire la voix d’une chanson. Nous disposons pour cela d’une base de données consistant en une multitude de morceaux de musique avec pour chaque morceau le canal du mix (voix + basse + batterie + autres) et celui de la voix uniquement. Cette base de données nous servira à entraîner un réseau de neurones convolutif avec pour entrée du réseau le signal mixé et en sortie la voix uniquement. Les signaux temporels sont passés dans le domaine spectral afin d’obtenir des spectrogrammes ; ce sont ces images qui forment les données d’entrée et de sortie du réseau.

Le schéma ci-dessous rend compte, de façon synthétique et à posteriori, des développements menés :

![trame](https://user-images.githubusercontent.com/77715595/148669344-5bd49060-9696-470c-bb2b-19dd412e5b4c.jpg)


## Mise en forme des données   
##### Notebook associé : DS21-P10-A-PERIMETRE

L’ensemble du projet repose sur l’exploitation de la base de données DSD100 (15 Go), disponible en open source, contenant 100 morceaux de musique avec pour chaque titre, 5 fichiers .wav contenant le morceau tel quel aussi appelé canal mix, le canal vocal pour la voix, bass pour la bass, drums pour les percussions et other pour le reste de l’accompagnement. Tous les fichiers sont originellement échantillonnés à 44100Hz.

Les données fournies par la base DSD100 (fichiers .wav ainsi qu’un fichier Excel précisant le style de musique de chaque morceau) nous ont permis de construire une dataframe utile pour l’exploration des morceaux et la dataviz.
Pour ce faire, nous avons également considérablement été aidés par le module complémentaire « RECONNAISSANCE VOCALE-CONCEPTS FONDAMENTAUX » et la librairie librosa pour l’écoute et le chargement des morceaux. La figure ci-dessous est une copie des premières lignes :

![df](https://user-images.githubusercontent.com/77715595/148669573-ca78aeb6-0f75-477e-8a9d-a216289ceacf.jpg)

Nous avons immédiatement constaté que ce jeu de données était très propre, dénué de bruit de fond ou d’artéfact sur l’ensemble des canaux.
Par la suite, nous nous rendrons compte que ce jeu de référence dans le domaine de la séparation de voix est à la fois d’une part limité en nombre de morceaux pour parvenir à un excellent résultat et, d’autre part, conséquent et lourd à exploiter par l’équipement informatique dont nous disposons. Le deep-learning est ici incontournable.

## Exploration des données et Dataviz
##### Notebook associé : DS21-P10-B-DATAVIZ

#### Verification sur signaux 
![donnee_propre](https://user-images.githubusercontent.com/77715595/148669645-3fb49449-9b99-452e-b957-8dc22f712777.jpg)

Le graphique de gauche rend compte de l‘amplitude en fonction du pas de temps pour chacun des canaux, autre que le mix, sur une plage de temps donnée.
Le graphique central rend compte de l’amplitude du signal mix et de la somme des précédant signaux.
Le graphique de droite rend compte du résultat de la soustraction des signaux du graphique central. Il y a bien parfaite  correspondance et ceci se vérifie par sommation des colonnes de la dataframe.


#### Ré-éechantillonnage et conséquences sur la qualité des signaux
![reech](https://user-images.githubusercontent.com/77715595/148669710-39b68063-08dc-4e15-8dc7-085c154f6e21.jpg)

Modifier l’échantillonnage de nos signaux est nécessaire pour diminuer le volume de données, les temps de calcul et espace de stockage requis.
Il a été procédé à un ré-échantillonage de 44100 Hz vers 11025 Hz via le module torchaudio. Le graphique  illustre les écarts  d’amplitude entre les 2 signaux (le signal à 11025Hhz s’est vu rajouter des zéros une valeur sur quatre afin de permettre une comparaison « visuelle »).
Il résulte de cette analyse le fait qu’il est possible de constater visuellement et auditivement un écart. Néanmoins, cet écart est suffisamment faible pour que l’ensemble des canaux soit toujours audible et discernable.

#### Du choix d'un type de spectrogramme
![spectro](https://user-images.githubusercontent.com/77715595/148669764-c89aec4c-2547-4dca-a692-b300b5103688.jpg)

Bien que le spectrogramme MEL offre davantage de contrastes, donc optimise le fonctionnement de l’algorithme de traitement d’images, il est probable que le rétroMEL soit problématique en terme de récupération de données car la matrice de passage de 513 à 128 pas de fréquences n’est pas inversible. Souvent, les équipes qui travaillent sur des applications en lien avec le son et qui exploitent les spectrogrammes mels avec regroupement de fréquence restent en mels et ne retournent pas vers l’audio.

## Abstract
![abstr](https://user-images.githubusercontent.com/77715595/148669822-30d97a50-dd00-49c9-a9d4-9b65482cbc40.jpg)

#### APPROCHE N° 1 : Résultat avec le Vocal isolator
##### Notebook associé : DS21-P10-C-Vocal Isolator
![viiiiiii](https://user-images.githubusercontent.com/77715595/148849170-9fb894a1-4fa8-4021-949c-d37a327490ac.jpg)

#### APPROCHE N° 2 : Résultat avec le VAD + VI
##### Notebook associé : DS21-P10-D-VAD + VI
![vadviiiiiii](https://user-images.githubusercontent.com/77715595/148849450-6c8dc901-2358-41cb-8f12-a5c6c7570be5.jpg)

#### APPROCHE N° 3 : Résultat avec le UNET
##### Notebook associé : DS21-P10-E-UNET
![image](https://user-images.githubusercontent.com/77715595/148849526-6d70e4e1-d4ee-4403-9489-3afa9199cf96.png)

#### APPROCHE N° 4 : Résultat avec le VI associé à un generator
##### Notebook associé : DS21-P10-F-VI avec generator
![bien](https://user-images.githubusercontent.com/77715595/148849756-287578e3-91ab-47b3-aaab-b4153c13c022.jpg)

### Complément et annexes:

#### VAD temporel selon DS21-P10-G-VAD Temporel
Une prédiction de musique chantée et non chantée qui repose uniquement sur le traitement du signal temporel -> résultat au même niveau de l'approche fréquencielle.

#### VAD + VI temporel selon D21-P10-H- VI temporel
Une tentative d'utiliser uniquement les signaux temporels pour extraire le canal vocal -> pas de résultat probant.
