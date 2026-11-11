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

## Accuracy plots

See [results/README.md](results/README.md) for saving Colab evaluation results
and plotting a confusion matrix and per-digit accuracy. From the repository root:

```bash
python examples/vae_single_image_memory/results/plot_accuracy.py work/vae_single_image_memory/evaluation.json
```

To diagnose the original Hopfield projection with training/validation loss
curves while reusing the cached expert bank and training cues, see the
[diagnostic instructions](results/README.md#hopfield-trainingvalidation-curves-using-your-original-checkpoint).

## Matched single-image augmented training

The baseline `train.py` remains available for comparison. The entry point
`train_augmented.py` reuses its expert checkpoint and trains a fresh Hopfield
projection on single-image augmented cues. Pass `--augmentation legacy` to repeat
the original strong, fixed-order augmentation below. The current default is
`--augmentation mild`, described in the next section. Both profiles use the same
`augment_image` function and five-update VAE2 procedure as `evaluate.py`, which
automatically reads the profile from the checkpoint. Expert parameters, initialization, projection architecture,
loss, replay order, optimizer steps and learning rate are unchanged.

By default, the number of training cues per digit matches the source checkpoint.
Each new cue uses a different original image sampled from that digit's original
90% training split. This is a sampled subset, not a cue for every training image.
For example, 668 baseline cues become 668 single-source-image cues. Override
with `--cues-per-digit N` only for a separate cue-budget experiment. No original
image is duplicated within the training cue set. The remaining 10% split supplies
disjoint augmented validation sources. It previously selected expert checkpoints,
but never updates or selects Hopfield weights. Default validation cue counts
match the original diagnostic's full-block count (at least one per digit);
`--validation-cues-per-digit N` can increase the validation sample size.

In Colab, first mount Drive and pull the branch, then run from the repository root:

```python
SOURCE = "/content/hopfield-layers/work/vae_single_image_memory/checkpoint.pt"
OUTPUT = "/content/drive/MyDrive/NEURO/hopfield_experiment2/augmented_cues"

!python -m examples.vae_single_image_memory.train_augmented --source "{SOURCE}" --output-dir "{OUTPUT}" --device cuda --augmentation legacy
```

The source checkpoint is preserved. Outputs include `checkpoint.pt`,
`validation_cues.pt`, and `hopfield_training.json` with all ten replay stages.
The report also scores the frozen block-trained baseline on the exact same
augmented training/validation cue sets before refitting. Curves are recorded
during this actual new run:

```python
!python examples/vae_single_image_memory/results/plot_hopfield_training.py "{OUTPUT}/hopfield_training.json"

from IPython.display import Image, display
display(Image(filename=f"{OUTPUT}/hopfield_stage_10.png"))
```

Test with the existing evaluator. Keep the same `--seed` and `--test-limit` as
the baseline (default seed 123) to use the same original images, order and latent
sampling seeds. Only the saved Hopfield projection differs at inference:

```python
!python -m examples.vae_single_image_memory.evaluate --checkpoint "{OUTPUT}/checkpoint.pt" --output-dir "{OUTPUT}/test_1000" --device cuda --test-limit 1000 --seed 123
!python examples/vae_single_image_memory/results/plot_accuracy.py "{OUTPUT}/test_1000/evaluation.json"
display(Image(filename=f"{OUTPUT}/test_1000/evaluation_accuracy.png"))
```

The 1,000 test images already inspected are a comparison set, not an untouched
final test. Use validation results for subsequent choices and reserve fresh test
images or an independently held-out set for a final estimate.

## Mild, shuffled augmentation

The second augmentation setting keeps 80 views and five updates but uses:

- 40 equally spaced rotations spanning -15 to +15 degrees (endpoints included).
- 20 horizontal and 20 vertical axis scales spanning 0.9 to 1.1. Values below
  one compress the digit; values above one stretch it.
- A seeded random permutation of all 80 views before splitting into five batches
  of 16. Geometry remains deterministic; each episode seed chooses the ordering.

Canvas size, centering, bilinear interpolation and black background are unchanged.
The shuffle uses its own local random generator and does not consume the VAE's
latent-noise generator. Training, validation and test cue generation all use the
same profile and episode-seed convention. Source-image selection and cue budgets
match a legacy run from the same source checkpoint and command-line counts.

The profile and explicit geometry are saved as `augmentation_profile` and
`augmentation_config` in checkpoints, training/validation artifacts and evaluation
reports. Old checkpoints without those fields always retain legacy augmentation.
There is no evaluation override that silently switches an old projection to mild
cues; train a new projection for the mild run. Expert VAEs are reused unchanged.

In Colab after updating the checkout, use the original block-trained expert bank
as the source and a separate output directory (mount Drive first):

```python
SOURCE = "/content/hopfield-layers/work/vae_single_image_memory/checkpoint.pt"
MILD = "/content/drive/MyDrive/NEURO/hopfield_experiment2/augmented_mild"

!python -m examples.vae_single_image_memory.train_augmented --source "{SOURCE}" --output-dir "{MILD}" --device cuda --augmentation mild
!python examples/vae_single_image_memory/results/plot_hopfield_training.py "{MILD}/hopfield_training.json"

from IPython.display import Image, display
display(Image(filename=f"{MILD}/hopfield_stage_10.png"))

!python -m examples.vae_single_image_memory.evaluate --checkpoint "{MILD}/checkpoint.pt" --output-dir "{MILD}/test_1000" --device cuda --test-limit 1000 --seed 123
!python examples/vae_single_image_memory/results/plot_accuracy.py "{MILD}/test_1000/evaluation.json"
display(Image(filename=f"{MILD}/test_1000/evaluation_accuracy.png"))
```
