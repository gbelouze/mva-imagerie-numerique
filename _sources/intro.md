# Image fusion with Guided Filtering

This is a report for a work done by us, Raphaël Rozenberg and Gabriel Belouze, on the article **Image fusion with guided filtering** {cite}`image-fusion`. This was the final project for the course [Introduction à l'imagerie numérique](https://perso.telecom-paristech.fr/gousseau/MVA/) (see also [Julie Delon's website](https://delon.wp.imt.fr/enseignement/mva-introduction-a-limagerie-numerique/)) from the [MVA master](https://www.master-mva.com/).

## Project description (🇫🇷)

Comme vu en cours, un certain nombre de facteurs influent sur la profondeur de champs en photographie : ouverture, focale, distance au sujet. Il est ainsi fréquent (en particulier en macro-photographie ou en microscopie) de ne pas avoir une profondeur de champs suffisante pour capturer de manière nette l'ensemble des éléments d'une scène.

Le papier ci-dessus (comme de nombreux autres) propose d'augmenter la profondeur de champs en combinant plusieurs images acquises avec des réglages de mise au point différents. L'idée principale est qu'en chaque point est retenue l'information provenant de l'image la plus nette en ce point. Une difficulté est alors de combiner les informations provenant de plusieurs images sans créer d'artefacts. Le papier ci-dessus propose pour ce faire de recourir à une décomposition de l'image en deux couches : détails et contenu grossier. Cette décompoition est effectuée au moyen d'un filtrage anisotrope. La méthode peut s'adapter à d'autres modalités de fusion d'images (par exemple la fusion d'expositions, voir le projet sur ce sujet).

Le but du projet est de comprendre le papier, coder la méthode (le code du filtrage est déjà disponible) et la tester sur des jeux d'images fournies.

Si vous le souhaitez, il est possible de créer son propre jeu de données, mais ceci représente un travail assez conséquent qui n'est pas demandé dans le cadre du projet.

## Project description (🇬🇧)

As seen in class, a number of factors influence depth of field in photography: aperture, focal length, distance to the subject. It is therefore common (especially in macro photography or microscopy) not to have a sufficient depth of field to capture all the elements of a scene clearly.

The above paper (like many others) proposes to increase the depth of field by combining several images acquired with different focus settings. The main idea is that at each point the information from the sharpest image at that point is retained. One difficulty is then to combine the information from several images without creating artefacts. The above paper proposes to do this by decomposing the image into two layers: detail and coarse content. This decomposition is performed by means of anisotropic filtering. The method can be adapted to other image fusion modalities (e.g. exposure fusion, see the project on this topic).

The aim of the project is to understand the paper, code the method (the filtering code is already available) and test it on provided image sets.

If you wish, it is possible to create your own dataset, but this is quite a lot of work and is not required for the project.
