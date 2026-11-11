# Digit-specific VAE parameter memories

This is a separate experiment. The previous Split MNIST MLP example and its
results are unchanged. It uses the existing reusable `hflayers.modern.ModernHopfield`
class, enforcing `update_steps_max=0`.

## Networks and files

| File | Responsibility |
| --- | --- |
| `vae.py` | Fully connected VAE, negative ELBO, flatten/restore parameters |
| `training.py` | Train each main VAE0 expert and restore its best validation checkpoint |
| `cues.py` | Reset VAE2 and Adam for each independent 50-batch training block; unknown-image cues |
| `data.py` | MNIST in [0,1], reproducible training/validation splits and loaders |
| `hopfield.py` | Separate Hopfield network and supervised cue-to-memory optimization |
| `train.py` | Sequentially build the ten-memory bank and replay all observed cue variants |
| `evaluate.py` | Randomly mixed, unknown test images; label-free adaptation and memory selection |
| `tests/test_vae_memory.py` | Architecture, block reset/isolation, best checkpoint, cue training, prediction tests |

All VAE instances use the same architecture (150,139 trainable parameters):

- Encoder: 784 → 85 ReLU → 85 ReLU, then separate five-dimensional mean and
  log-variance heads.
- Latent variable: five-dimensional Gaussian, trained by reparameterization.
- Decoder: 5 → 85 ReLU → 85 ReLU → 784 pixel logits.
- Loss: pixel-summed binary cross entropy plus KL(q(z|x) || N(0,I)), averaged
  across images. Decoder probabilities are sigmoid(logits).

VAE0 is the **role of the main expert**, not a ten-class classifier. There are
ten independent VAE0 experts, trained in digit order 0,...,9. VAE2 is a separate
instance of the same architecture used only to construct parameter cues.
Hopfield is an independent neural network, not a layer in the VAE and not
jointly trained through the VAE loss. The initial parameters are shared as a
starting point, but no trained VAE parameters or optimizer state carry over
between experts or between VAE2 blocks.

## Memories and variant cues

For digit d, split its official MNIST training examples into 90% training and
10% validation. Train its VAE0 for ten epochs with Adam (lr=0.001). Select the
lowest validation negative-ELBO checkpoint as the single stored memory theta_d*.
Validation uses fixed latent noise across epochs. Save the parameters after
exactly the first **50 optimizer updates**, theta_d_first, as a reference cue;
this is not an end-of-first-epoch checkpoint.

Separately shuffle that digit's training examples and partition them into full
blocks. A block contains **50 minibatches** of **16 images**, or 800 distinct
images by default. For each block j:

1. Construct a fresh VAE2 from the common initial parameters.
2. Construct a fresh Adam optimizer with no previous momentum/variance state.
3. Train one epoch over that block: each image is visited once, exactly 50 updates.
4. Save theta_d_first_j and pair it with the same fixed memory theta_d*.

Incomplete final blocks are discarded for cue generation; they remain available
for main VAE0 training. Within a pass, blocks are disjoint. `--cue-passes` can
produce additional reshuffled variants, with overlap across passes explicitly
allowed. No validation or test images enter cue generation. k varies by digit:
with batch size 16 and one pass it is floor(N_train / 800).

After each arriving digit, train Hopfield using all accumulated variants and
memories (explicit cue replay). Training minimizes the cross entropy of the
correct memory association plus normalized parameter reconstruction MSE. The
VAE0 memories and VAE2 cues are fixed tensors; only the Hopfield encoder is
optimized. `main_first50_cues` are saved for inspection, while the actual
Hopfield training examples are the independently generated **VAE2 variants**.

The retriever standardizes using the memory bank, projects 2,048 deterministic
parameter coordinates into 64 association dimensions, and performs one cosine
softmax association (beta=20). It outputs a weighted combination of the full
150,139-coordinate memories. Use `--feature-size 150139` to project every
coordinate instead. No iterative Hopfield updates are performed.

## Unknown-image prediction

A test image does not supply a label or a task ID to either network. Since one
image does not itself supply 50 minibatches, the implemented inference protocol
repeats that image into a batch of the training batch size, then performs
**50 unlabeled VAE2 updates** from the common initialization with fresh Adam.
The VAE reconstruction/KL loss needs no digit label.

Feed the resulting parameter cue into Hopfield with all ten memories. The
highest association's stored digit is the predicted digit. The soft retrieved
parameter vector is loaded into another VAE to reconstruct the image. A VAE has
no ten-class output head; digit classification here means **memory selection**.

Training blocks contain many different images of one digit, whereas a test cue
uses one repeated unknown image. This is a substantial distribution shift;
correct retrieval on training variants does not establish unknown-image
accuracy. Evaluation measures this explicitly rather than assuming it works.
The default evaluation is a balanced random subset of 100 test images (ten per
digit), shuffled together. Labels are used for subset allocation and metrics,
never inference. Each image is processed independently; there are no
class-homogeneous support episodes and no true-label memory routing.

## Run

From the repository root:

```bash
python -m pip install -r examples/vae_parameter_memory/requirements.txt
python -m unittest discover -s examples/vae_parameter_memory/tests -v
python -m examples.vae_parameter_memory.train
python -m examples.vae_parameter_memory.evaluate
```

The checkpoint and JSON reports default to `work/vae_parameter_memory/`.
`checkpoint.pt` contains the common initialization, ten full memory vectors,
Hopfield weights, VAE2 cue vectors and targets, main first-50 cues, exact block
indices, configuration, and memory digit labels. `training.json` records each
expert's best epoch/validation losses and replay retrieval accuracy.
`evaluation.json` records unknown-image accuracy, confusion matrix, per-digit
accuracy, associations, selected memories, and posterior-mean reconstruction
BCE. `reconstructions.pt` saves reconstructed pixels and original test indices.
The reconstruction metric uses the posterior mean, not a stochastic ELBO.

For all 10,000 test images (50 VAE2 updates **per image**, potentially expensive):

```bash
python -m examples.vae_parameter_memory.evaluate --test-per-digit 0
```

A smaller real-MNIST integration run, retaining two variants per digit:

```bash
python -m examples.vae_parameter_memory.train --epochs 2 --train-limit 2000 \
  --hopfield-steps 50 --output-dir work/vae_memory_smoke
python -m examples.vae_parameter_memory.evaluate \
  --checkpoint work/vae_memory_smoke/checkpoint.pt \
  --output-dir work/vae_memory_smoke --test-per-digit 2
```

## Recorded run

The committed `results/` files contain a full-data ten-epoch training run
(seed 42) and evaluation on 100 shuffled, balanced test images (seed 123).
There are ten memories of 150,139 parameters and 62 VAE2 training cues:
`[6,7,6,6,6,6,6,7,6,6]` for digits 0 through 9. All five new unit tests pass.
Reloading the saved checkpoint gives 100% retrieval accuracy on those training
variants. It also selects the correct memory for all ten saved VAE0
first-50-batch reference cues, which were not used as Hopfield training inputs.

Unknown-image memory-selection accuracy is **39/100 (39%)**. Per-digit numbers
of correct predictions out of ten are `[8,10,4,2,6,0,5,2,0,2]`.
Thus, successful retrieval from multi-image training blocks does **not** imply
reliable retrieval from a single repeated image. This result is retained
without tuning on test labels. Better generalization would require a separate
study of training cues that match the intended inference input. The trained
checkpoint and reconstructions remain under `work/`, outside source control;
the committed reports contain the configuration and individual predictions.
