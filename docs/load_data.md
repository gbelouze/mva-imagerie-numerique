---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: imnum
  language: python
  name: imnum
---

```{code-cell} ipython3
:tags: [remove-cell]

# modules
from imports import np, plt, iio, cv2, gaussian, Path, gf, filters, data, fusion, recalage

# functions
from imports import gray_to_rgb, plot_images

# variables
from imports import BASE_DIR, multi_exposure_dataset, multi_focus_dataset, ours_dataset, multi_exposure_sample, multi_exposure_sample, multi_focus_sample, ours_sample, gray_sample
```

+++ {"id": "203a5efb-11bf-4588-82d9-3daa9cd8f2a2", "tags": []}

# Preliminaries

+++ {"id": "bd3ec6c6-fe1a-4065-a851-902a41ce8568", "tags": []}

## Load data

+++ {"id": "TkGIrJwap9OL"}

**Datasets we used:**

- *Petrovic* {cite}`petrovic` : grayscale multi-focus dataset, by pairs of pictures

- *Lytro* {cite}`lytro` : color multi-focus dataset, mostly by pairs

- *MEFDatabase* {cite}`multiexposure` : color multi-exposure dataset, several pictures each time

- Our own pictures [Gabriel Belouze]: multi-focus, several pictures each time

```{code-cell} ipython3
:id: 76ce6734-cd83-4311-aee4-d7af0ff7733e
:tags: ["hide-input"]

multi_exposure_dataset = data.MultiviewDataset(Path(f"{BASE_DIR}/data/MEFDatabase/source/"))
multi_focus_dataset = data.MultiviewDataset(Path(f"{BASE_DIR}/data/lytro"))
ours_dataset = data.MultiviewDataset(Path(f"{BASE_DIR}/data/ours"))

multi_exposure_sample = multi_exposure_dataset["Balloons_Erik Reinhard"]
multi_exposure_sample = multi_exposure_dataset["Lighthouse_HDRsoft"]
multi_focus_sample = multi_focus_dataset["20"]
ours_sample = ours_dataset["stylos2"]


gray_sample = [iio.imread(path) / 255 for path in [f"{BASE_DIR}/data/petrovic/input001_{i}.tif" for i in (1, 2)]]
```

+++ {"id": "658a456c-1c8c-4ee7-b81e-26d049bfb04a"}

## Show data


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 195
id: cf55f8d0-ad5d-4fa7-ae3e-c27d4afe98d7
outputId: a00c96ed-4a69-4683-bdde-7c4b0c52ce4c
render:
  figure:
    caption: 'Multi-exposure sample {cite}`multiexposure`'
    name: multi-exposure-sample
  image:
    width: 600px
---
plot_images(*multi_exposure_sample)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 384
id: 2617f237-fc63-47d6-ac0d-68adf71b98cc
outputId: 09bf4cb4-883b-4e4f-c768-748a854d8c06
render:
  figure:
    caption: 'Multi-focus sample {cite}`lytro`'
    name: multi-focus-sample
  image:
    width: 600px
---
plot_images(*multi_focus_sample)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 500
id: 4NqBXo9euZqI
outputId: a0ed79d6-499c-4ba0-d20b-a7034d90d92d
render:
  figure:
    caption: 'Our own multi-focus sample'
    name: own-sample
  image:
    width: 600px
---
plot_images(*ours_sample, maxwidth=4)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 384
id: Jy94A6Ksuayu
outputId: 4b0b696e-db6a-4f03-c2b9-17514cd96f46
render:
  figure:
    caption: 'Gray sample {cite}`petrovic`'
    name: gray-sample
  image:
    width: 600px
---
plot_images(*gray_sample)
```
