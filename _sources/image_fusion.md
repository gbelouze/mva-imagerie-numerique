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

+++ {"id": "3ff8f33a-c73d-4c5d-a118-c83463a49cbe", "tags": []}

# Image fusion with guided filtering

+++

The method aims at combining several images, either in a multi-focus setting or in a multi-exposure setting.

```{tip}
:class: margin
**Saliency** is defined as the local average of the norm of the image Laplacian.
```


The images are combined through a weighted sum. The weights are determined according to the saliency map of each image, which measures the level of spatial variation at each pixel. In addition, the method aims at leveraging structural information, especially by preserving edges as much as possible thanks to guided filtering.

This method can be used indifferently for multi-focus and multi-exposure problems because in both cases, the saliency map is a relevant quantity. Indeed, in the multi-focus setting, images are salient where they are sharp, whereas in the multi-exposure setting they are salient where they are well-exposed.

+++ {"id": "1b846fbf-5c5a-4e7b-8351-704dd15bfa12"}

![image_fusion_schematic.png](images/image_fusion_schematic.png)

+++ {"id": "7a1e9f2c-4e0f-45fc-bbfd-9d20123b086a", "tags": []}

## Step by step

```{code-cell} ipython3
:id: Ju6jGIx1D4yR
:tags: []

gff_focus = fusion.gff(multi_focus_sample)
```

+++ {"id": "82e8a561-a006-4257-913a-ce2c4847fc4a"}

### Weight map
Weight maps are constructed to be $1$ at pixel $i$ if the image has the highest saliency (i.e. gradient norm) at pixel $i$, and $0$ otherwise.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 409
id: 0982b868-2ad5-4940-9d8c-2ae63301534b
outputId: b26f2643-aa49-47d0-8d57-15455418fee2
tags: ["hide-input"]
render:
  figure:
    caption: 'Weight maps'
    name: weight-maps
  image:
    width: 600px
---
plot_images(*gff_focus.weights, 
            labels=['weight map 1', 'weight map 2'])
plt.show()
```

+++ {"id": "7794d5c1-19f4-44d9-a9ab-da8a4ab1f2ac"}

### Base / Detail decomposition
Then, images are split into a base layer and a detail layer. Each layer will have its own weight map and be fused back in only at the very end.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 859
id: 7e448666-d324-4f07-8d12-27890a44ee3f
outputId: 9e9b2802-0e78-4a60-9443-3689b074d4b3
render:
  figure:
    caption: 'Base layers'
    name: base-layers
  image:
    width: 600px
tags: ["hide-input"]
---
base_layer, detail_layer = zip(*[gff_focus.decompose(im) for im in gff_focus.ims])
plot_images(*base_layer)
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 859
id: 7e448666-d324-4f07-8d12-27890a44ee3f
outputId: 9e9b2802-0e78-4a60-9443-3689b074d4b3
render:
  figure:
    caption: 'Detail layers'
    name: detail-layers
  image:
    width: 600px
tags: ["hide-input"]
---
plot_images(*detail_layer)
plt.show()
```

+++ {"id": "3dc96065-26da-4e2e-91dc-4c3fd513391a"}

### Refined weight map
The key idea is to use guided filtering with the original image as guidance. This mitigates noise and edge-aligns weight maps. 

Different parameters (see below) are used to get distinct weight maps for the base and details layers. In general, a larger window size, and a larger regularisation $\varepsilon$ are used for the base layer.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 802
id: 23a7c709-6f54-4d1d-a8e7-9b451752d2ee
outputId: ae0b8506-07c2-4abc-fb19-9a4890f7eb0b
tags: ["hide-input"]
render:
  figure:
    caption: 'Refined weight maps (base layer)'
    name: refined-weight-maps-base
  image:
    width: 600px
---
fused_focus = gff_focus.fusion()
plot_images(*gff_focus.refined_weights["base"])
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 802
id: 23a7c709-6f54-4d1d-a8e7-9b451752d2ee
outputId: ae0b8506-07c2-4abc-fb19-9a4890f7eb0b
render:
  figure:
    caption: 'Refined weight maps (detail layer)'
    name: refined-weight-maps-detail
  image:
    width: 600px
tags: ["hide-input"]
---
plot_images(*gff_focus.refined_weights["detail"])
plt.show()
```

+++ {"id": "d62f2ad2-925b-41e6-b94d-4f68485eee05"}

### Fusion
Those refined weight maps are used to fuse images in each layer base and details). Finally, the layers are added up to produce a single final image.  
For multi-focus, the base fusion is not that important as the base images basically look the same (a blurry version of the scene). However, for HDR, both base fusion and detail fusion must be done carefully.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 335
id: Bv4d0PNQeWdp
outputId: 52e85af5-0aa1-49e7-98f2-f241b74cba24
tags: ["hide-input"]
render:
  figure:
    caption: 'Fusion'
    name: __fusion
  image:
    width: 800px
---
plot_images(gff_focus.fused["base"], gff_focus.fused["detail"], fused_focus, 
            labels=['fused bases', 'fused details', 'final result'])
plt.show()
```

+++ {"id": "70cccba6-1302-43b4-a458-3ffb6655d90c"}

## Examples

+++ {"id": "6nFG63-fa5RT"}

Because the concept of *saliency* discriminates both in-focus against out-of-focus, and saturated against non-saturated, the method above can actually be used both for HDR and multi-focus recomposition.

```{code-cell} ipython3
:id: 0d88155b-6698-466d-97f8-39318ea0a1ab

exposure_gff = fusion.gff(multi_exposure_sample)
multi_exposure_fused = exposure_gff.fusion(r1=100, r2=15)

focus_gff = fusion.gff(multi_focus_sample)
multi_focus_fused = focus_gff.fusion()

gray_gff = fusion.gff(gray_sample)
multi_gray_fused = gray_gff.fusion()
```

```{code-cell} ipython3
---
render:
  figure:
    caption: 'Multi exposure fusion'
    name: exposure-fusion
  image:
    width: 800px
tags: ["remove-input"]
---
plot_images(*multi_exposure_sample, multi_exposure_fused, 
            labels=['input 1', 'input 2', 'input 3', 'result'])
plt.show()
```

```{code-cell} ipython3
---
render:
  figure:
    caption: 'Multi focus fusion'
    name: focus-fusion
  image:
    width: 800px
tags: ["remove-input"]
---
plot_images(*multi_focus_sample, multi_focus_fused, 
            labels=['input 1', 'input 2', 'result'])
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 743
id: bba2f39f-e5ab-43ba-9d03-5240c4b6260d
outputId: 7606eea0-0f21-489b-ceb2-2771c4fc8f6d
render:
  figure:
    caption: 'Gray fusion'
    name: gray-fusion
  image:
    width: 800px
tags: ["remove-input"]
---
plot_images(*gray_sample, multi_gray_fused, 
            labels=['input 1', 'input 2', 'result'])
plt.show()
```

+++ {"id": "AJEtxOCCJLY8"}

## Managing out-of-bounds values

+++ {"id": "k5w3NTKvJF61"}

The authors do not speak about how to handle out of bounds values. Out-of-bounds may appear during two steps:

1. When applying the **guided filters**, there is no guarantee that the output will stay either in the input image or guidance image bounds. It seems reasonable to clip the filtered weights between 0 and 1, otherwise we cannot normalize the weights in a meaningful way.

2. When **recombining** the base and detail layer, we can also get out-of-bounds values. Simply rescaling the output may noticeably "grayify" the image if there are values in the output too far out of bounds. We chose to also clip the image instead.
