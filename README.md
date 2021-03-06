# mva-imagerie-numerique

## How to use this repository ?

You can [read the report online](https://gbelouze.github.io/mva-imagerie-numerique/). If you want to run it yourself, we suggest to
- create a python virtual environment, for instance named `imagerie-numerique`
- install the dependencies
```bash
pip install -r requirements.txt
pip install -e .
```
- install your virtual environment as a jupyter kernel
```bash
pip install ipykernel
python -m ipykernel install --user --name=imagerie-numerique
```
When you open the notebook, `imagerie-numerique` should be available as a kernel.

## Articles

### Main

1. [Image fusion with guided filtering](https://perso.telecom-paristech.fr/gousseau/MVA/Projets2021/FocusFusion/fusion.pdf) Li, S., Kang, X., & Hu, J. (2013).

### Other resources

2. [Guided Filter](http://kaiminghe.com/publications/eccv10guidedfilter.pdf) He, Sun, Tang (2010)

## Descriptif

Comme vu en cours, un certain nombre de facteurs influent sur la profondeur de champs en photographie : ouverture, focale, distance au sujet. Il est ainsi fréquent (en particulier en macro-photographie ou en microscopie) de ne pas avoir une profondeur de champs suffisante pour capturer de manière nette l'ensemble des éléments d'une scène.

Le papier ci-dessus (comme de nombreux autres) propose d'augmenter la profondeur de champs en combinant plusieurs images acquises avec des réglages de mise au point différents. L'idée principale est qu'en chaque point est retenue l'information provenant de l'image la plus nette en ce point. Une difficulté est alors de combiner les informations provenant de plusieurs images sans créer d'artefacts. Le papier ci-dessus propose pour ce faire de recourir à une décomposition de l'image en deux couches : détails et contenu grossier. Cette décompoition est effectuée au moyen d'un filtrage anisotrope. La méthode peut s'adapter à d'autres modalités de fusion d'images (par exemple la fusion d'expositions, voir le projet sur ce sujet).

Le but du projet est de comprendre le papier, coder la méthode (le code du filtrage est déjà disponible) et la tester sur des jeux d'images fournies.

Si vous le souhaitez, il est possible de créer son propre jeu de données, mais ceci représente un travail assez conséquent qui n'est pas demandé dans le cadre du projet.
