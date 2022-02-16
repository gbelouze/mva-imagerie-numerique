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

# Experiments and remarks

+++

Our analysis mostly boils down to [ablation study](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence))

+++ {"id": "3ihiCoA_kY64"}

## Parameters and their impact

+++ {"id": "yeojR01cpSiX"}

The free parameters of the fusion method are the window size and regularization for the two guided filters (for the base and detail layer, respectively). The choice of parameters can be crucial, especially for multi-exposure fusion. This is less the case for multi-focus fusion.

The article is contrasted about this issue, stating at first:

> "the GFF
method does not depend much on the exact parameter choice."

However, the article ends with 

> "adaptively choosing the parameters of the guided
filter can be further researched."

which indicates that the question cannot be entirely dismissed, especially for multi-exposure fusion.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 221
id: P1_q3fb63dYM
outputId: d7424ae6-7266-4a49-e379-3580d2bbc998
tags: ["remove-input"]
render:
  figure:
    caption: 'Input images'
  image:
    width: 600px
---
multi_exposure_sample = multi_exposure_dataset["Lighthouse_HDRsoft"]
plot_images(*multi_exposure_sample,
            labels=['input 1', 'input 2', 'input 3'])
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: s9eid5nalxup
outputId: ebec86b9-87b9-44f4-bba9-a12d8fd4fe34
render:
  figure:
    caption: 'Influence of the base layer parameters'
  image:
    width: 600px
tags: ["hide-input"]
---
exposure_gff = fusion.gff(multi_exposure_sample)

multi_exposure_fused = exposure_gff.fusion(r1=100, r2=15, eps1=0.3, eps2=1e-6)
multi_exposure_fused_1 = exposure_gff.fusion(r1=3, r2=15, eps1=0.3, eps2=1e-6)
multi_exposure_fused_2 = exposure_gff.fusion(r1=1000, r2=15, eps1=0.3, eps2=1e-6)
multi_exposure_fused_3 = exposure_gff.fusion(r1=100, r2=15, eps1=1e-6, eps2=1e-6)
multi_exposure_fused_4 = exposure_gff.fusion(r1=100, r2=15, eps1=100, eps2=1e-6)

plot_images(multi_exposure_fused,
            labels=['good fusion'])

plot_images(multi_exposure_fused_1, multi_exposure_fused_2,
            labels=['smaller window (base layer)', 'bigger window (base layer)'])

plot_images(multi_exposure_fused_3, multi_exposure_fused_4,
            labels=['less regularization (base layer)', 'more regularization (base layer)'])
plt.show()
```

+++ {"id": "bjnXMy3BsG2m"}

Modifying the value of either $r$ (the window size) or $\epsilon$ (the regulaization parameter) in **the base layer parameters** induces important defects.

- Top row (window size $r$): **patches appear when $r$ is too small**, and many zones are too light when $r$ is too big (averaging on wide zones).


- Bottom row (regularization parameter $\epsilon$): **incoherent result with insufficient regularization**: the house seems flat, the beach is much lighter than the sea, etc. However increasing regularization has little effect.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: iOuAjdybr2MU
outputId: 14166af8-5270-4be9-ffa8-35f6eab8a137
render:
  figure:
    caption: 'Influence of the detail layer parameters'
  image:
    width: 600px
tags: ["hide-input"]
---
multi_exposure_fused_1 = exposure_gff.fusion(r1=100, r2=1, eps1=0.3, eps2=1e-6)
multi_exposure_fused_2 = exposure_gff.fusion(r1=100, r2=500, eps1=0.3, eps2=1e-6)
multi_exposure_fused_3 = exposure_gff.fusion(r1=100, r2=15, eps1=0.3, eps2=1e-10)
multi_exposure_fused_4 = exposure_gff.fusion(r1=100, r2=15, eps1=0.3, eps2=1e2)

plot_images(multi_exposure_fused,
            labels=['good fusion'])

plot_images(multi_exposure_fused_1, multi_exposure_fused_2,
            labels=['smaller window (detail layer)', 'bigger window (detail layer)'])

plot_images(multi_exposure_fused_3, multi_exposure_fused_4,
            labels=['less regularization (detail layer)', 'more regularization (detail layer)'])
plt.show()
```

+++ {"id": "IKl-eKOPn6_d"}

Most changes are less noticeable when toying with the **details layer parameters**, probably because the layer is more sparse.

- Top row ($r$) : once again, patches appear when $r$ is too small (even though less noticeably), but more importantly **details are lost when $r$ is too big**.


- Bottom row ($\epsilon$): no noticeable changes.

+++ {"id": "mYmvfC0ZBdKf"}

## Is guided filtering necessary?

+++ {"id": "UPW7q7fVBpzn"}

The authors suggest guided filtering as a filter method which knows about the structure of the (source/guide) image, and specifically its edges. We investigated how useful this method was against:
- no-filtering
- a naive gaussian filter

+++ {"id": "ZD_aEWgQeD7O"}

### No filtering

+++ {"id": "6d04OVuleCPw"}

It is necessary to perform some kind of smoothing, in order to prevent patches to appear. This can be easily seen in the case of exposure fusion:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 89
id: iQ6dBJshf_xH
outputId: 6f3c9500-43a1-4d29-a7bb-58851706f086
render:
  figure:
    caption: 'No filtering'
  image:
    width: 700px
tags: ["hide-input"]
---
# no weights filtering
fused_exposure = exposure_gff.fusion(filt=lambda p, *args, **kwargs: p)
plot_images(*multi_exposure_sample)
plot_images(*exposure_gff.normalised_refined_weights["base"]) #same as detail

plot_images(fused_exposure)
```

+++ {"id": "X95QWhcZeHXN"}

### Gaussian filtering

+++ {"id": "XhnmX_68n550"}

What about using a gaussian filter instead of a guided filter that leverages the structure of a guidance image?

```{code-cell} ipython3
:id: LELkG57rhewL

def gaussian_filter(p,i,r,eps):
  return gaussian(p, r)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 54
id: wwPkZYEcqIVu
outputId: 37b674d0-2a60-4f12-c0d8-42da2ce64b53
---
multi_exposure_sample = multi_exposure_dataset["Memorial_Debevec97"]
exposure_gff = fusion.gff(multi_exposure_sample)
plot_images(*multi_exposure_sample, maxwidth=4)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 72
id: mxfp0K6zoQ4g
outputId: f0184496-0fb1-4228-dbf9-7eaca1e00058
---
# Guided filter with source image as guide
fused_exposure = exposure_gff.fusion(r1=20, eps1=1e-2)

plot_images(*exposure_gff.normalised_refined_weights["base"],
            title="Guided filter: Refined weights [Base layer]",
            maxwidth=4)

plot_images(*exposure_gff.normalised_refined_weights["detail"],
            title="Guided filter: Refined weights [Detail layer]",
            maxwidth=4)
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 72
id: 4Dw6On9iizUi
outputId: 75d54cef-f5bf-4f05-865f-a61c3b512d93
---
# Gaussian filter
fused_exposure_gauss = exposure_gff.fusion(filt=gaussian_filter,
                                           r1=10)

plot_images(*exposure_gff.normalised_refined_weights["base"],
            title="Gaussian filter: Refined weights [Base layer]",
            maxwidth=4)

plot_images(*exposure_gff.normalised_refined_weights["detail"],
            title="Gaussian filter: Refined weights [Detail layer]",
            maxwidth=4)
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 54
id: RpVbbDBtAUV_
outputId: c6970ee4-01e5-453f-f420-fd2388bee636
---
plot_images(fused_exposure_gauss, fused_exposure,
            labels=['Gaussian filtering fusion', 'Guided filtering fusion'])
```

+++ {"id": "fxD4p7s-jUAe"}

Gaussian filtering yields less satisfying results than guided filtering, especially in the most difficult zones (here around the stained glasses).

However, it is **much** faster, so in terms of real life applications it might be more useful.

+++ {"id": "Q62N1WhqBys3"}

## Base/detail separation

+++ {"id": "hiOllLr0pZlu"}

The base / detail separation is only useful to use different guidance parameter for the guided filtering. However, as shown *figure 8* of the article, the dependance to those parameters seems minimal.

Here we show that in most cases (but not all), it seems that the base/detail separation is actually not necessary. In several examples below, the results are extremely similar (in fact indistinguishable to the naked eye) when using or not base/details separation:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 72
id: rMTm1Zs_uhOv
outputId: ad31fe96-e109-4f22-b518-f073b15c98dd
---
multi_focus_sample = multi_focus_dataset["20"]
focus_gff = fusion.gff(multi_focus_sample)
multi_focus_fused = focus_gff.fusion()
multi_focus_fused_no_sep = focus_gff.fusion_without_separation(r=20)


plot_images(*multi_focus_sample,
            labels=['input 1', 'input 2'])


plot_images(multi_focus_fused, multi_focus_fused_no_sep,
            labels=["Base-Detail separation", "No separation"])
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 54
id: ICAZR0NkBrNQ
outputId: 7d0ce474-984b-4dbd-ae41-81ea91b6570e
---
multi_exposure_sample = multi_exposure_dataset["Memorial_Debevec97"]
plot_images(*multi_exposure_sample, maxwidth=4,
            title = "input pictures")
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 54
id: jm4yaGXinrk3
outputId: 95459abe-f792-4712-9a5a-915e02def5e3
---
exposure_gff = fusion.gff(multi_exposure_sample)

multi_exposure_fused = exposure_gff.fusion(r1=20, eps1=1e-2)
multi_exposure_fused_no_sep = exposure_gff.fusion_without_separation(r=20, eps=1e-2)

plot_images(multi_exposure_fused, multi_exposure_fused_no_sep,
            labels=["Base-Detail separation", "No separation"])
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 72
id: 9AmJd8hSmzSU
outputId: 9cc2e6f6-32a7-47ba-8a50-6431d3b7c281
---
multi_exposure_sample = multi_exposure_dataset["Lighthouse_HDRsoft"]
exposure_gff = fusion.gff(multi_exposure_sample)

multi_exposure_fused = exposure_gff.fusion(r1=100, r2=15)
multi_exposure_fused_no_sep = exposure_gff.fusion_without_separation(r=100)

plot_images(*multi_exposure_sample,
            labels=["input 1", "input 2", "input 3"])



plot_images(multi_exposure_fused, multi_exposure_fused_no_sep,
            labels=["Base-Detail separation", "No separation"])
plt.show()
```

+++ {"id": "0XN2g-kasUeq"}

However, sometimes it is impossible to find a satisfying ($r, \epsilon$) combination. The base/details separation allows to use distinct ($r, \epsilon$) pairs for each layer.

```{code-cell} ipython3
:id: s8Fs0ElTo_27
:tags: []

multi_exposure_sample = multi_exposure_dataset["Cadik Lamp_Martin Cadik"]
plot_images(*multi_exposure_sample)
exposure_gff = fusion.gff(multi_exposure_sample)


multi_exposure_fused = exposure_gff.fusion(r1=80, eps1=1e-1, r2=5, eps2=1e-3)
multi_exposure_fused_no_sep_1 = exposure_gff.fusion_without_separation(r=80, eps=1e-1)
multi_exposure_fused_no_sep_2 = exposure_gff.fusion_without_separation(r=20, eps=1e-1)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 72
id: Fpax0rARpflu
outputId: 06d87740-b0ca-4f00-9767-5839fc31fe8c
---
plot_images(multi_exposure_fused, multi_exposure_fused_no_sep_1,
            labels=["Base-Detail separation", "No separation [1]"])

plot_images(multi_exposure_fused, multi_exposure_fused_no_sep_2,
            labels=["Base-Detail separation", "No separation [2]"])
plt.show()
```

+++ {"id": "QJU88oc_tdkZ"}

Here, **without** base/details separation (right side), we either have a halo around the lamp or dark patches on the sheet of paper and on the walls.

+++ {"id": "lFBKHKnJqPes"}

## Other color spaces: HSV

+++ {"id": "wmSOWoP-VDvO"}

The method works just as well with color spaces other than RGB, for example HSV, despite the doubtful relevance of doing linear combinations of the coefficients in the energy formula (and despite the discontinuity in the Hue channel).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 183
id: 9TAokqtqUIT-
outputId: 31b3aa19-0fff-417d-e935-90716fcd4f05
---
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
multi_exposure_sample = multi_exposure_dataset["Lighthouse_HDRsoft"]

# data: rgb to hsv
hsv_input = [rgb_to_hsv(im) for im in multi_exposure_sample]

# gff
hsv_gff = fusion.gff(hsv_input, color='hsv')
fused_hsv = hsv_gff.fusion()

# result: hsv to rgb
result = hsv_to_rgb(fused_hsv)
plot_images(*multi_exposure_sample, result, 
            labels=['input 1', 'input 2', 'input 3', 'result'],
            title="Fusion with HSV channels")
plt.show()
```

+++ {"id": "-6izHhUWVpNY"}

By examining the application of the guided filter, we see that sharp edges remain as long as there is the V channel in the guide. The H channel is also necessary in order to avoid color artefacts.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 678
id: 8Pwn4jApVod9
outputId: 95042d23-adef-4fed-f7fb-980fdb68f347
---
im_hsv = hsv_input[1]
h, s, v = im_hsv[:,:,0], im_hsv[:,:,1], im_hsv[:,:,2]

plot_images(h, s, v, labels=["H", "S", "V"])

out_v = hsv_to_rgb(filters.guided_filter(im_hsv, v, r=7, eps=1e-3))
out_h = hsv_to_rgb(filters.guided_filter(im_hsv, h, r=7, eps=1e-3))
out_full = hsv_to_rgb(filters.guided_filter(im_hsv, im_hsv, r=7, eps=1e-3))

plot_images(out_v, out_h, out_full, 
            labels=['V guide only', 'H guide only', 'full guide'],
            title="Guided filtering (HSV space)")
```

+++ {"id": "nqy6v0ENaO9-"}

By contrast, on a standard image, all three RGB channels roughly share the same edges thus yield satisfying results as the guide.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 678
id: koRsvycvZjnD
outputId: fcea4036-76b8-4f33-9d29-1309482a0f12
---
im_rgb = multi_exposure_sample[1]
r, g, b = im_rgb[:,:,0], im_rgb[:,:,1], im_rgb[:,:,2]

plot_images(r, g, b, labels=["R", "G", "B"])


out_r = filters.guided_filter(im_rgb, r, r=7, eps=1e-3)
out_g = filters.guided_filter(im_rgb, g, r=7, eps=1e-3)

out_full = filters.guided_filter(im_rgb, im_rgb, r=7, eps=1e-3)

plot_images(out_r, out_g, out_full, 
            labels=['R guide only', 'G guide only', 'full guide'],
            title="Guided filtering (RGB space)")
```
