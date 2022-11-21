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

## Description

As seen in class, a number of factors influence depth of field in photography: aperture, focal length, distance to the subject. It is therefore common (especially in macro photography or microscopy) not to have a sufficient depth of field to capture all the elements of a scene clearly.

The above paper (like many others) proposes to increase the depth of field by combining several images acquired with different focus settings. The main idea is that at each point the information from the sharpest image at that point is retained. One difficulty is then to combine the information from several images without creating artefacts. The above paper proposes to do this by decomposing the image into two layers: detail and coarse content. This decomposition is performed by means of anisotropic filtering. The method can be adapted to other image fusion modalities (e.g. exposure fusion, see the project on this topic).

The aim of the project is to understand the paper, code the method (the filtering code is already available) and test it on provided image sets.

If you wish, it is possible to create your own dataset, but this is quite a lot of work and is not required for the project.
