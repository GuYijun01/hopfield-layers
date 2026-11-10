# Split MNIST parameter memories

Run from the repository root:

```bash
python -m pip install -r examples/split_mnist/requirements.txt
python -m examples.split_mnist.experiment --output-dir work/split_mnist
python -m unittest discover -s tests -v
```

The experiment learns five memories in order: (0,1), (2,3), (4,5), (6,7),
(8,9). Each memory contains all 478,410 parameters of a 784–400–400–10 MLP
with ReLU hidden activations. Adam minimizes ten-class cross entropy; softmax
probabilities can be obtained from the returned logits. Each expert starts from
one shared, seeded initialization so training and test adaptation cues are in
the same parameter coordinate system. This is sequential **expert-memory**
learning, rather than carrying one MLP's weights across task boundaries.

For each task, 90% of its official training examples train the expert; 10%
select the best epoch by validation cross entropy. The restored best checkpoint
is the memory. The parameters immediately after the first epoch form its
training cue. N is the number of minibatches in that epoch, including a final
partial batch. Hopfield trains on all observed cue/memory pairs after every
new task (explicit replay). It minimizes association cross entropy plus
normalized parameter reconstruction error. The memory values remain fixed.

`ModernHopfield` accepts a batch of flattened cues and a memory bank. A learned
shared encoder projects a deterministic selection of parameter coordinates into
an association space, using memory-bank standardization and cosine similarity.
One softmax with beta=20 retrieves a weighted combination of the **full**
parameter vectors. The default 2,048 selected coordinates avoid a huge dense
478,410-dimensional projection; `--feature-size 478410` uses every coordinate.
The association width defaults to 64. `update_steps_max=0` is enforced: there
are no iterative retrieval updates. The class is independent of the older
`HopfieldCore` implementation.

## Evaluation protocol

The user-requested labeled test-time adaptation is explicit. Each task's
shuffled official test examples are divided equally into labeled support and
disjoint query examples. A fresh MLP starts from the shared initialization,
with a fresh Adam optimizer, and adapts on support data. Support is cycled if
N exceeds its number of batches. N/2 means floor(N/2), with a minimum of one
update; both budgets use the same sequence and optimizer trajectory.

Both cues retrieve from the bank of all seen memories, without supplying a
task ID to Hopfield. The retrieved parameters are loaded into the MLP; query
cross entropy and accuracy use all ten output classes, without a task mask.
Metrics are collected for every seen task after each arriving task. The data
sampler provides task-homogeneous support/query episodes; this assumption and
the use of support labels make this an **adaptation-assisted Class-IL**
experiment, not a standard unlabeled, mixed-stream Class-IL benchmark.

`results.json` reports per-task direct-cue and retrieved loss/accuracy,
association weights, selected memory, N, best epoch, validation history, and
mean loss differences (N minus N/2; negative favors N). Means weight tasks
equally. `checkpoint.pt` saves the initialization, Hopfield weights, memory
bank, train/test cues, configuration, and task order. Test data never trains
Hopfield or selects expert checkpoints. Hyperparameters are not selected using
query results.

A quick run on real MNIST, with explicit sample caps:

```bash
python -m examples.split_mnist.experiment --epochs 2 --train-limit 1000 \
  --test-limit 200 --hopfield-steps 50 --output-dir work/split_mnist_smoke
```

For a full run, omit the sample caps. The default is ten expert epochs, five
tasks, batch size 64, and 200 Hopfield optimizer steps per task. A single seed
is a functional experiment; use multiple `--seed` values for scientific claims.

## Recorded full-data run

`results_seed42.json` records the default full-data run (seed 42, ten epochs,
all five tasks). Final task-mean query metrics after learning all memories:

| Adaptation budget | Retrieved cross entropy | Retrieved accuracy |
| --- | ---: | ---: |
| N/2 | 0.008685329592 | 99.7610% |
| N | 0.008685329946 | 99.7610% |

Both budgets select the corresponding memory for all five task episodes.
The mean N-minus-N/2 loss difference is approximately 3.54e-10, so this run
provides no evidence of a retrieval advantage from the extra updates. Inspect
the direct-cue metrics in the JSON to distinguish adaptation quality from
retrieval saturation. These numbers apply to the adaptation-assisted protocol
above and should not be compared directly to standard Class-IL results.
