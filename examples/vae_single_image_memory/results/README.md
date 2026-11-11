# Accuracy plots and evaluation results

`plot_accuracy.py` reads an evaluation JSON and plots a row-normalized confusion
matrix with counts plus per-digit recall. The title reports overall per-image
accuracy. Each original image counts once, regardless of its 80 augmentations.
Digits without test samples are marked N/A. No measured Colab results have
been added yet.

## Colab: save your existing evaluation and plot in this folder

Update the checkout first, then copy the actual evaluator output from Drive.
The evaluator names its output `evaluation.json`; the sample below renames the
copy to identify the run. Adjust the source path to match your evaluation's
`--output-dir`, and use a distinct run name for different checkpoints or seeds.

```python
%cd /content/hopfield-layers
!git pull origin heterogeneous-memory

from pathlib import Path
import shutil

source = Path("/content/drive/MyDrive/NEURO/hopfield_experiment2/test_1000/evaluation.json")
destination = Path("examples/vae_single_image_memory/results/colab_1000_seed123")
destination.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination / "evaluation.json")

!python examples/vae_single_image_memory/results/plot_accuracy.py "{destination}/evaluation.json"

from IPython.display import Image, display
display(Image(filename=str(destination / "evaluation_accuracy.png")))
```

If the source is missing, locate it with
`list(Path("/content/drive/MyDrive/NEURO").rglob("evaluation*.json"))`.
Mount Google Drive first. Matplotlib and NumPy are normally installed in Colab;
otherwise install the experiment's requirements.

The JSON and PNG are then inside the local repository. To publish those two
files, use your authenticated GitHub session, commit their exact paths, and
push to `heterogeneous-memory`. Never put access tokens in a notebook cell.
Avoid adding checkpoint or per-image cue files, which are large and unnecessary
for these plots.

## Hopfield training/validation curves using your original checkpoint

The original training code saved final projection weights and one summary per
digit, not optimizer-step histories. Those historical curves cannot be recovered.
The diagnostic first evaluates the exact saved projection, then starts a fresh
projection and repeats sequential replay on cached training cues. It does not
retrain expert VAEs or change the cue protocol. The original checkpoint is never
overwritten. Default steps and learning rate come from that checkpoint.

Validation cues use disjoint full 80-image blocks from each digit's original
10% validation split of the official MNIST training set. Each cue uses five
updates with fresh VAE2 weights and Adam. These images never enter expert or
Hopfield gradient training, but were used to select the expert checkpoints;
this is a Hopfield holdout diagnostic, not a completely untouched end-to-end
validation set. No official test images enter these diagnostics. Full-data
checkpoints have enough validation images; heavily capped smoke checkpoints
may not. At least 80 validation images per digit are required.

From Colab after pulling this branch:

```python
CHECKPOINT = "/content/hopfield-layers/work/vae_single_image_memory/checkpoint.pt"
DIAGNOSTICS = "/content/drive/MyDrive/NEURO/hopfield_experiment2/hopfield_diagnostics"

!python -m examples.vae_single_image_memory.diagnose_hopfield --device cuda --checkpoint "{CHECKPOINT}" --output-dir "{DIAGNOSTICS}"
!python examples/vae_single_image_memory/results/plot_hopfield_training.py "{DIAGNOSTICS}/hopfield_training.json"

from IPython.display import Image, display
display(Image(filename=f"{DIAGNOSTICS}/hopfield_stage_10.png"))
```

Adjust `CHECKPOINT` if you saved it to Drive. Mount Drive first. The report
contains exact frozen-original metrics plus histories for all ten replay stages.
Plots default to the ten-memory stage; pass `--stage 5` for five memories, etc.
Each stage has a different memory bank, so compare train/validation within a
stage rather than connecting stages as one curve. Stage one is trivially perfect.

Four panels show the actual total objective, association CE, normalized parameter
MSE, and correct-memory accuracy. Total loss is CE plus parameter MSE divided by
the full memory bank's population variance. The old summary's `loss` field only
reported CE; it was not the full training objective. Measurements include step
zero and follow optimizer updates, with validation evaluated without gradients.
Use `--log-every 5` to reduce logging overhead. Validation never selects weights
or controls early stopping. The diagnostic saves `validation_cues.pt` and a
separate `refit_checkpoint.pt`, as well as the JSON report.

Low training loss with high validation loss suggests poor generalization even
for the original distinct-image block protocol. Low losses on both suggest the
single-image test mismatch merits a separate investigation. These curves do not
measure augmented-single-image performance.
