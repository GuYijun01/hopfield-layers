# Experiment 2: five-update cues and single-image test adaptation

This separate experiment reuses the original VAE architecture, negative ELBO,
expert training, MNIST loaders, and Hopfield retriever. Expert VAE0 training
remains ten epochs by default, batch size 16, with validation-best checkpoints.
All experts and cue VAEs share one initialization. There is no weight or Adam
state carryover between cue episodes.

Training cues use five full batches of 16 distinct training images (80 per
block). Blocks are digit-homogeneous and disjoint within each pass; tails are
unused. Each cue is theta_after_5 minus theta_init. Sequential Hopfield replay,
delta keys, full-parameter outputs, and one softmax association are unchanged.
The main expert's first-five-update delta is saved for inspection only.

At test time, each official test image independently generates 80 variants:
40 equally spaced rotations in [-pi/4, pi/4], including endpoints; 20 horizontal
stretches; and 20 vertical stretches. Stretch factors range from 1.05 to 1.5.
Transforms are centered, use bilinear interpolation, retain a 28x28 canvas,
fill with black, and crop content outside the canvas. The original image is
not an extra 81st sample. The fixed order is rotations, horizontal stretches,
vertical stretches, partitioned into five batches of 16 and visited once.
A fresh VAE2 and Adam produce a delta cue after exactly five updates.
No label enters augmentation, VAE2, or Hopfield inference. Labels only score
retrieval. Every source image is processed independently in shuffled order;
all ten memories compete. Accuracy is per source image. Source-image
reconstruction BCE is an adaptation metric, not held-out reconstruction.
Training cues use distinct images while test cues use one image's augmentations;
this deliberate distribution change needs measurement, and no accuracy is claimed.

## Google Colab

From the repository root, with a GPU runtime selected:

```python
!python -m examples.vae_single_image_memory.train --device cuda
!python -m examples.vae_single_image_memory.evaluate --device cuda --test-limit 100
```

Omit `--test-limit 100` to evaluate all 10,000 test images (50,000 VAE updates).
For CPU use `--device cpu`. Both training and evaluation use batch size 16.
A short full-data-path smoke run can use `--epochs 1 --train-limit 100
--hopfield-steps 2` during training. This is not the default scientific run.

Outputs default to `work/vae_single_image_memory/`: `checkpoint.pt`,
`training.json`, and `evaluation.json`. Copy these to Google Drive before Colab
resets. Old 50-update checkpoints are rejected: train a new checkpoint here.
Optional `--save-cues` streams a separate cue/full-parameter file per image;
this costs about 12 GB for the complete test set, so it is off by default.

```bash
python -m unittest discover -s examples/vae_single_image_memory/tests -v
```
