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
