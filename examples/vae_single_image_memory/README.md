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

## Same training images, different cue seeds

`diagnose_seeds.py` freezes an existing augmented-cue Hopfield checkpoint and
uses exactly its stored training source indices, targets and source order.
It neither resamples images nor updates Hopfield or expert weights. It compares:

1. The cached training cues, scored by the saved projection.
2. Cues regenerated from the same images using their original episode seeds.
3. Five repeated regenerations with new episode seeds by default.

Every new cue still starts from the same initialization with fresh Adam and uses
80 views in five batches of 16. Default `--vary both` changes the view permutation
and VAE latent noise; geometry stays fixed. `--vary shuffle` changes only view
ordering, and `--vary latent` changes only latent sampling. The legacy profile
has fixed ordering, so it supports only `latent` or `both`.

This diagnostic evaluates the final frozen network; the horizontal axis is a
seed condition, not an optimizer step. It records the exact training objective
(association CE plus normalized parameter MSE), both component losses, accuracy,
prediction agreement with cached cues, relative cue change and per-image results.
Scoring uses small chunks (default 64 cues) to reduce memory use without changing
the objective's equal weighting over source images. Minor floating-point
differences from full-batch reductions are possible.

For Colab, keep input/output on local storage to avoid mounted-Drive checkpoint
I/O errors. If necessary, copy the mild checkpoint back from Drive using
`shutil.copyfile`. From `/content/hopfield-layers` after pulling the branch:

```python
CHECKPOINT = "/content/augmented_mild_run/checkpoint.pt"
STABILITY = "/content/augmented_mild_seed_diagnostics"

!python -m examples.vae_single_image_memory.diagnose_seeds --checkpoint "{CHECKPOINT}" --output-dir "{STABILITY}" --device cuda --runs 5
!python examples/vae_single_image_memory/results/plot_seed_stability.py "{STABILITY}/seed_stability.json"

from IPython.display import Image, display
display(Image(filename=f"{STABILITY}/seed_stability.png"))
```

Original seeds are read from `cue_seeds` in newer checkpoints or reconstructed
from the existing `train_augmented.py` seed schedule for older ones; no retraining
is needed. Same-seed replay should closely match cached cues and loss. If it does
not, first check the dataset, software and device used for reproduction before
attributing differences to new seeds. A substantial loss increase with new seeds
would indicate sensitivity to the stochastic cue-generation trajectory. Stable
performance here alongside poor validation suggests the gap concerns unseen
source images instead. Seed runs are repeated paired measurements on the same
images, not independent new training or validation examples.

To isolate the two sources of randomness, repeat the command with `--vary shuffle`
and `--vary latent`, using separate output directories. No large cue files are
saved by this diagnostic; JSON and PNG can be copied to Drive afterward.

## Balanced transforms, per-update shuffle, and more source images

New files `balanced_cues.py` and `train_balanced.py` live in this same package.
No additional source or output directories are required. Prior experiments
remain available. The new 80-view pool has five equally sized families:

| Family | Views | Values |
| --- | ---: | --- |
| Rotation | 16 | Equally spaced from -10 to +10 degrees |
| Horizontal scale | 16 | Equally spaced from 0.90 to 1.10 |
| Vertical scale | 16 | Equally spaced from 0.90 to 1.10 |
| Uniform scale | 16 | Both axes together, equally spaced from 0.90 to 1.10 |
| Translation | 16 | All x/y combinations of {-2, -1, +1, +2} pixels |

Each family transforms the original image independently. Transforms are centered,
use bilinear interpolation and black background, and retain a 28x28 canvas.
Translation limits are per axis, so diagonal displacement can exceed two pixels.

Before **each** of the five VAE2 updates, make a fresh random permutation of the
entire pool and take its first 16 views. There are no duplicate indices within a
batch. Views can repeat across batches, and some pool members can be unused.
This is per-update resampling, not a single shuffled pass over all 80 views.
The same seed-controlled procedure runs in training, validation and testing.
VAE2 parameters and Adam state reset for every source image, with the same
negative-ELBO objective and five updates. No new expert VAE training is required.

Defaults are **500 distinct training sources per digit (5,000 cues)** and
**100 validation sources per digit (1,000 cues)**, sampled from the original
90/10 split. Override `--images-per-digit N` (`0` uses every training source)
and `--validation-per-digit N`. A sample-capped expert checkpoint must have
enough images in its original splits for the requested counts. Validation images
previously selected the expert checkpoints but never update or select Hopfield.

Hopfield still uses the shared 2048-to-64 projection, all seen memories, full-batch
optimizer steps, and the same CE plus normalized parameter-MSE loss. For memory
deltas A, association weights alpha and one-hot targets y, the MSE numerator is
`(alpha-y) @ (A @ A.T / D) @ (alpha-y).T` per cue. The implementation uses the
small memory Gram matrix rather than materializing a B-by-D retrieved tensor.
Only the selected cue coordinates need caching because the original projection
uses those coordinates. The full expert memories are preserved for inference.
Tests compare both loss and encoder gradients against explicit full-parameter
retrieval. This is an algebraic computation change, not a new model or loss.

### Colab

After mounting Drive and updating the repository in your separate setup cells,
restore a completed expert or mild checkpoint to local storage. Reuse the
`CHECKPOINT` variable from your restoration cell (for example,
`/content/augmented_mild_run/checkpoint.pt`). Use prefixes in existing directories; commands do not create
experiment folders:

```python
%cd /content/hopfield-layers
!git pull origin heterogeneous-memory

PREFIX = "/content/balanced80"
BACKUP = "/content/drive/MyDrive/NEURO/hopfield_experiment2/balanced80"

!python -m examples.vae_single_image_memory.train_balanced --source "{CHECKPOINT}" --output-prefix "{PREFIX}" --backup-prefix "{BACKUP}" --device cuda
```

The local files are `balanced80_cue_cache.pt`, `balanced80_checkpoint.pt` and
`balanced80_training.json`. The backup prefix creates the same filenames in your
existing `hopfield_experiment2` folder, without a new subfolder. Cue caches are
saved and optionally backed up after each completed digit. On interruption,
restore the saved cache to the local prefix and rerun the same command; completed
cue generation is reused after checking its settings and expert-bank fingerprint.
Hopfield optimization restarts from the same seeded initialization using the
cache. Prefixes with incompatible existing caches are rejected.

```python
!python examples/vae_single_image_memory/results/plot_hopfield_training.py "{PREFIX}_training.json" --output "{PREFIX}_training.png"

from IPython.display import Image, display
display(Image(filename=f"{PREFIX}_training.png"))

!python -m examples.vae_single_image_memory.evaluate --checkpoint "{PREFIX}_checkpoint.pt" --output-prefix "{PREFIX}" --device cuda --test-limit 1000 --seed 123
!python examples/vae_single_image_memory/results/plot_accuracy.py "{PREFIX}_evaluation.json" --output "{PREFIX}_accuracy.png"
display(Image(filename=f"{PREFIX}_accuracy.png"))
```

The evaluator selects the balanced per-update sampler from checkpoint metadata;
older checkpoints keep their original sampler. Balanced checkpoints cache
`cue_features` rather than full `cues` and are not inputs to the older
`diagnose_seeds.py`. Full VAE parameters are still retrieved at test time.
Plot and evaluation files can be copied to Drive using `shutil.copyfile` after
completion. No accuracy improvement is assumed before measurement.

### Two-phase Hopfield learning rate

To use 600 updates per trained digit stage, with updates 1–300 at 0.001 and
301–600 at 0.0003, pass `--hopfield-steps 600 --hopfield-lr 0.001
--lr-drop-after 300 --hopfield-lr-final 0.0003`. The rate changes in place:
Adam moments and its step counter continue through the boundary. The schedule
restarts for each arriving digit, with a fresh Adam optimizer as before.
The trivial one-memory stage still skips optimizer updates.

The schedule is stored in the JSON and checkpoint. History records the rate
used for each logged update and always includes updates 300 and 301. The plot
marks the rate switch. Without these two new schedule flags, training retains
its constant-rate behavior.

Reuse the exact existing cue cache with a different output prefix to preserve
your original results. No expert training or cue regeneration is needed. After
updating the repository, run (restore these two input files locally if needed):

```python
SOURCE = "/content/balanced80_checkpoint.pt"
CACHE = "/content/balanced80_cue_cache.pt"
SCHEDULED = "/content/balanced80_600"
BACKUP = "/content/drive/MyDrive/NEURO/hopfield_experiment2/balanced80_600"

!python -m examples.vae_single_image_memory.train_balanced --source "{SOURCE}" --cue-cache "{CACHE}" --output-prefix "{SCHEDULED}" --backup-prefix "{BACKUP}" --device cuda --hopfield-steps 600 --hopfield-lr 0.001 --lr-drop-after 300 --hopfield-lr-final 0.0003
!python examples/vae_single_image_memory/results/plot_hopfield_training.py "{SCHEDULED}_training.json" --output "{SCHEDULED}_training.png"

from IPython.display import Image, display
display(Image(filename=f"{SCHEDULED}_training.png"))
```

Defaults assume the previous 500 training and 100 validation sources per digit.
If your original cache used different source counts, pass those same counts.
As in other balanced runs, optimization starts from the shared seeded projection
and replays all digits. It does not append updates to the old final checkpoint.
Files are written under new prefixed names in existing directories.

### Separate query and key projections

Pass `--projection-mode separate` to learn independent Wq and Wk. Each maps
the same selected 2,048 coordinates into 64 association dimensions (262,144
learned weights total by default). Queries use Wq and memory-delta keys use Wk,
followed by the existing unit normalization and beta=20 cosine softmax. The full
memory deltas remain the values; no value or output projection is introduced.

Wq and Wk start with the same seeded values as the shared baseline but have
independent parameter storage and Adam moments. Their gradients differ and they
can diverge during training. This preserves starting predictions while removing
the weight-sharing constraint. `--projection-mode shared` remains the default
for previous commands. Checkpoints record the mode and evaluation automatically
constructs the correct architecture. Older checkpoints default to shared mode.

Reuse the existing balanced cue cache and the same 600-update learning-rate
schedule. This retrains the projection stages from the seeded initialization;
it does not continue optimization from the old final projection. No expert VAE
training or cue regeneration is needed. No new folders are created:

```python
%cd /content/hopfield-layers
!git pull origin heterogeneous-memory

SOURCE = "/content/balanced80_checkpoint.pt"
CACHE = "/content/balanced80_cue_cache.pt"
SEPARATE = "/content/balanced80_qk"
BACKUP = "/content/drive/MyDrive/NEURO/hopfield_experiment2/balanced80_qk"

!python -m examples.vae_single_image_memory.train_balanced --source "{SOURCE}" --cue-cache "{CACHE}" --output-prefix "{SEPARATE}" --backup-prefix "{BACKUP}" --projection-mode separate --device cuda --hopfield-steps 600 --hopfield-lr 0.001 --lr-drop-after 300 --hopfield-lr-final 0.0003
!python examples/vae_single_image_memory/results/plot_hopfield_training.py "{SEPARATE}_training.json" --output "{SEPARATE}_training.png"

from IPython.display import Image, display
display(Image(filename=f"{SEPARATE}_training.png"))

!python -m examples.vae_single_image_memory.evaluate --checkpoint "{SEPARATE}_checkpoint.pt" --output-prefix "{SEPARATE}" --device cuda --test-limit 1000 --seed 123
```

Restore the two input files from Drive if local storage was reset. Separate
outputs preserve the previous shared-projection checkpoint and loss curves.
