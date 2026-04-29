# Part 1.6 — Texture Segmentation

We split each image into regions based on local texture.

## How it works

1. Build a Gabor filter bank: 4 orientations times 3 wavelengths = 12
   filters. Each filter responds strongly to texture going in a
   specific direction at a specific scale.
2. For every pixel, run all 12 filters and take the absolute value
   (we want the strength of the response, not its sign). Then smooth
   each filter response so the feature is stable, not jumpy.
3. (Color version only) Add the Lab `a` and `b` channels to the
   feature vector. Lab is the color space where distance matches what
   the eye sees.
4. Standardize the features (z-score) and run K-Means with k=4.
5. Color in each pixel by which cluster it belongs to.

## What we save per image

- `Grayscale_Texture_Images/<weather>__<name>__seg_k4.png`,
  using only the Gabor features. The assignment asks us to try
  grayscale first, so this is that.
- `Color_Texture_Images/<weather>__<name>__seg_k4.png`,
  Gabor + Lab(a, b). The assignment asks us to add color after, so
  this is that.

Each output is a side-by-side strip: the original image on the left,
the segmentation on the right.

## What we noticed

- Sky, road, buildings, and vegetation usually separate well even in
  grayscale, because they have very different Gabor patterns.
- Adding color helps tell apart things that look similar in texture
  but different in color, like grass vs. dirt or lane markings vs.
  asphalt.
- Fog and snow images have less to work with. The clusters tend to
  collapse into fewer real groups, which we mention more in Part 3.
