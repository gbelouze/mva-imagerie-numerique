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

+++ {"id": "9e2890ad-772a-4def-9532-f6a3b0517793", "tags": []}

# Guided filtering

+++ {"id": "c3c3671e-2ce9-47f3-a478-7d0b152bf5fa"}

![guided_filter_schematic.png](images/guided_filter_schematic.png)

+++

Guided filtering {cite}`guided-filter` is a smoothing method which tries to respect image structure (and most notably edges). In this regard, it resembles bilateral filtering. However, guided filtering is able to preserve edges of *another* image.  
Guided filtering takes two images, $P$ and $I$ as inputs and produces one image $O$ as output.
- $P$ is called the input image. $O$ is designed to look like $P$.
- $I$ is called the guidance image. $O$ is structured like $I$ in that it should share the same edges.

Concretely speaking, to produce $O_i$ the output at pixel $i$, we look at $w_i$ a window centered at $i$. We try to construct an affine transformation of the corresponding window in $I$, so that the window produces looks like the corresponding window in $P$. This means that we minimize over $a_i$ and $b_i$ the energy
\begin{equation*}
E(a_i, b_i) = \sum_{k \in w_i} ( a_i I_k + b_i - P_k)^2 + \varepsilon a_i^2
\end{equation*}
where we added a regularisation term $\varepsilon a_i^2$.

There are some additional details :
- if $I$ is RGB, we look at $\mathbf{a}_i^T \mathbf{I}_i$ instead
- if $P$ is RGB, we realize 3 filtering for the 3 $P$ channels

In our case, $I$ will be RGB and $P$ gray.

+++ {"id": "1ed50094-f880-4e6e-8bf2-6b96cbb45880", "tags": []}

## Gray guide

```{code-cell} ipython3
:id: 00a5a736-5d24-46ea-a33a-dd583f79d47d

input = multi_focus_sample[0].mean(axis=-1)
guide = multi_focus_sample[1].mean(axis=-1)
output = filters.guided_filter(input, guide, r=20, eps=5e-2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 297
id: 1db043dc-d7c4-4c02-845a-4ef61b96e3cf
outputId: 33fdd8ee-f681-4879-fd3b-7becadce28fd
tags: ["remove-input"]
---
plot_images(input, guide, output, 
            labels=['input 1', 'input 2', 'output'])
```

+++ {"id": "2f2321a2-1883-4c55-a096-e9ee2a4b3582"}

Notably, the edges of the buildings in the backgrounds remain sharp.

+++ {"id": "45a230a1-df3d-4d5f-8d43-22f7cdcb7651", "tags": []}

## RGB guide

```{code-cell} ipython3
:id: 698d176a-6c84-49bc-8b19-82c802cc119f

input = multi_focus_sample[0]
guide = multi_focus_sample[1]
output = filters.guided_filter(input, guide, r=20, eps=5e-2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 297
id: 7fcea546-0a9d-4e6f-8a5f-a4ae4b9ce886
outputId: 2c56d894-350b-4130-f3e7-dbb02afbca63
tags: ["remove-input"]
---
plot_images(input, guide, output, 
            labels=['input', 'guide', 'filter output'])
```
