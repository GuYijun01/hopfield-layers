# Digit-specific VAE parameter memories

This experiment is separate from the previous Split MNIST MLP example. The
VAE and Hopfield networks have separate parameters and training objectives.
Hopfield performs one association, enforcing `update_steps_max=0`.

## Networks and files

| File | Responsibility |
| --- | --- |
| `vae.py` | VAE architecture, negative ELBO, flatten/restore parameters |
| `training.py` | Train ten independent VAE0 experts and select best validation checkpoints |
| `cues.py` | Reset VAE2 and Adam for each 50-batch block; return parameter-change cues |
| `data.py` | MNIST in [0,1], training/validation splits and loaders |
| `hopfield.py` | Train the Hopfield encoder; retrieve full VAE parameters from delta cues |
| `train.py` | Build expert memories and replay training-cue variants sequentially |
| `refit.py` | Convert cached legacy cues to deltas and refit without retraining VAE0 |
| `evaluate.py` | Retrieve memories from held-out test-image blocks |
| `tests/test_vae_memory.py` | Six focused tests, including delta origins and full-parameter restoration |

Every VAE has 150,139 trainable parameters:

- Encoder: 784 → 85 ReLU → 85 ReLU → separate 5-dimensional mean/log-variance heads.
- Gaussian latent variable: five dimensions, trained by reparameterization.
- Decoder: 5 → 85 ReLU → 85 ReLU → 784 pixel logits; sigmoid gives pixel probabilities.
- Loss: pixel-summed BCE plus KL(q(z|x) || N(0,I)), averaged over images.

VAE0 denotes the main-expert role. Ten independent experts are trained in digit
order 0,...,9, each from the same initial parameters. VAE2 is a separate instance
of the same architecture, freshly initialized from those shared parameters for
every block. No trained VAE weights or Adam state carry between blocks/experts.
Hopfield optimization does not backpropagate into either VAE.

## Training memories and delta cues

For each digit c, 90% of its official MNIST training examples train its VAE0;
10% select the best checkpoint among ten epochs. Adam uses lr=0.001. Fixed
latent noise makes validation comparisons reproducible. The one best **full**
parameter vector theta_c* becomes that digit's stored memory.

Shuffle the training examples into disjoint full blocks. With batch size
**d=16**, each block contains **N=50 minibatches**, or **800 images**. A fresh
VAE2 and fresh Adam optimizer visit each block once (one epoch over the block).
After exactly 50 updates, the cue is

    q_c,j = vec(theta_c,j^(50)) - vec(theta_init).

This is a parameter **difference**, not the raw partially trained weights.
`fit_block` implements the subtraction:

```python
return parameter_vector(model) - initial_vector(initial_state).to(device)
```

For digit c, multiple block cues q_c,1,...,q_c,k all target the same memory
theta_c*. k=floor(N_train/800) by default. Incomplete tails are unused for cue
generation, but remain in VAE0 training. `--cue-passes` reshuffles for additional
variants; blocks do not overlap within a pass, but can overlap across passes.
Validation/test images never enter these training cues.

After each arriving digit, Hopfield trains with replay of all observed training
cue/memory pairs. Its loss combines correct-memory association cross entropy
with normalized parameter reconstruction MSE. Only the Hopfield encoder is
optimized. The main VAE0's first-50 **delta** cue is saved separately for
inspection; Hopfield training uses the VAE2 block variants.

Both query and key coordinates use the same origin: memory keys are
vec(theta_c*)-vec(theta_init). The original full memories remain stored. The
retrieved delta receives the shared initialization back before loading into a
VAE. Thus the output contains full weights and biases, not a delta mistakenly
interpreted as network parameters. See `retrieve_parameters`.

The shared learned projection uses 2,048 deterministic coordinates and 64
association dimensions, cosine similarities, and beta=20 softmax. Retrieval
returns a weighted combination of the full memories. `--feature-size 150139`
uses every parameter coordinate. There are no iterative Hopfield updates.

## Held-out test-block retrieval

Use official MNIST **test** images, not training images, to form test cues.
For a block, reset VAE2 and Adam to the same starting state, train on its first
50 full minibatches once, and compute the same q_c^(50) difference vector.
Feed that cue to the frozen Hopfield network with all ten stored memories.
The highest association identifies the retrieved memory/digit. No label or
memory index enters `predict_block`, VAE2's loss, or Hopfield inference.

The benchmark assumes **digit-homogeneous blocks** (the meaning of q_c). The
evaluator uses test labels only to construct these benchmark blocks and score
retrieval. Blocks arrive in shuffled digit order. This is identification of one
unknown digit's block; it does not classify individual images in an arbitrary
mixed-digit block, which would not have a single correct target memory.

Use nonoverlapping test blocks, without repeating one image or cycling samples.
With d=16, all ten digits provide one complete 800-image block: **10 test cues,
8,000 adaptation images**. The remaining **2,000** test images are disjoint
query images for reconstruction scoring using the retrieved full parameters.
The main result is accuracy over test **blocks**, not 8,000 independent digit
predictions. No test cue, test label, or query metric trains/selects Hopfield.

## Run

From the repository root:

```bash
python -m pip install -r examples/vae_parameter_memory/requirements.txt
python -m unittest discover -s examples/vae_parameter_memory/tests -v
python -m examples.vae_parameter_memory.train
python -m examples.vae_parameter_memory.evaluate
```

Outputs default to `work/vae_parameter_memory_delta/`. Checkpoints carry
`cue_representation="parameter_delta_v1"`; evaluation rejects older raw-cue
checkpoints rather than silently mixing representations.

To reuse an existing VAE bank and cached training blocks from the earlier run:

```bash
python -m examples.vae_parameter_memory.refit \
  --source work/vae_parameter_memory/checkpoint.pt
python -m examples.vae_parameter_memory.evaluate
```

Refitting subtracts the initialization from cached raw training cues, starts a
fresh Hopfield encoder, and replays digits sequentially. It changes neither
the stored VAE0 memories nor their validation-selected checkpoints. No test
data enters this conversion. Newly trained checkpoints already use deltas.

`checkpoint.pt` saves memories, initialization, delta training cues, targets,
main first-50 delta cues, block indices, Hopfield weights, and configuration.
`training.json` records training/refit results. `evaluation.json` records each
test block, associations, selected memory, full-parameter RMSE against its
correct stored memory, and reconstruction BCE on disjoint query images.
`test_cues.pt` saves test delta cues, retrieved full parameters, and exact
adaptation/query indices. Query BCE uses posterior-mean decoding, not a
stochastic ELBO. `--blocks-per-digit` can cap full test blocks; zero uses all.

## Historical results

`results/legacy_single_image/` preserves the earlier raw-parameter,
repeated-single-image evaluation (39/100). It is a different protocol and is
not the current test-block experiment. Its source is available at commit
`e397c6a`. New delta/block reports are stored separately in `results/`.

## Recorded delta/block run

`results/delta_training_seed42.json` records refitting on the 62 cached training
block cues from the unchanged ten-epoch VAE0 experts. The checkpoint conversion
was verified against the original raw cues, and expert memories are bitwise
unchanged. A separate from-scratch integration run also exercised the new
training path (one epoch, 900 examples per digit, one cue per digit).

`results/delta_blocks_seed123.json` records **10/10 correct memory selections**
on ten held-out, digit-homogeneous test blocks, presented in shuffled order.
Every block contains 800 distinct images and exactly 50 updates. The 8,000
adaptation images and 2,000 reconstruction-query images are disjoint and cover
the official 10,000-image test set. This is ten block decisions, not 10,000
independent classifications. Mean parameter RMSE of the soft retrieved output
against the correct full memory is approximately **0.00209839**; selecting the
correct memory does not mean its soft output is numerically identical to that
memory. Saved test cues reproduce the recorded associations after checkpoint
reload. All six unit tests pass.
