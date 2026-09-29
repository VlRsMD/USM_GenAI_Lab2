# Lab 2: Classic Autoencoder vs. Variational Autoencoder on MNIST

**Goal:** implement, train and compare two encoder–decoder architectures, a
classic **Autoencoder (AE)** and a **Variational Autoencoder (VAE)**, on MNIST
for compressing, reconstructing and generating images.

## Project structure

```
Lab2/
├── main.py                      # CLI: train / compare / run
├── requirements.txt
├── autoencoders/
│   ├── config.py                # dataclass configs (data, model, training, paths)
│   ├── device.py                # CUDA probe + CPU fallback (runs before torch loads)
│   ├── utils.py                 # seeding, checkpoints, JSON I/O
│   ├── losses.py                # BCE reconstruction, KL divergence, AE / VAE losses
│   ├── data/
│   │   └── mnist.py             # train / validation / test DataLoaders
│   ├── models/
│   │   ├── base.py              # EncoderDecoder interface (encode, decode, reconstruct, generate)
│   │   ├── components.py        # shared convolutional encoder and decoder
│   │   ├── autoencoder.py       # classic AE
│   │   ├── vae.py               # VAE with the reparameterization trick
│   │   └── factory.py           # build a model by name ("ae" / "vae")
│   ├── training/
│   │   ├── trainer.py           # model-agnostic training loop, keeps best validation weights
│   │   └── pipeline.py          # build model + optimizer, train, save checkpoint and history
│   ├── evaluation/
│   │   ├── metrics.py           # test-set BCE, MSE, PSNR, KL, latent statistics
│   │   ├── latent.py            # PCA projection, interpolation, 2-D latent grid
│   │   └── compare.py           # runs every metric and figure for both models
│   └── visualization/
│       └── plots.py             # matplotlib figures
├── tests/
│   └── test_models.py           # shape, loss and helper sanity checks
└── outputs/                     # created on run: checkpoints/, figures/, metrics/
```

Both models share the **same convolutional backbone** (`components.py`), so any
difference in behaviour comes from what differs between them: the latent head
and the loss.

## The two models

| | Autoencoder | Variational Autoencoder |
|---|---|---|
| Encoder output | one vector `z = f(x)` | a distribution `q(z|x) = N(μ(x), σ²(x))` |
| Latent code during training | deterministic | sampled: `z = μ + σ·ε`, `ε ~ N(0, I)` (reparameterization trick) |
| Loss | reconstruction (BCE) | reconstruction (BCE) + β·KL(q(z|x) ‖ N(0, I)) (negative ELBO) |
| Latent space | unconstrained; codes can sit anywhere | pulled toward the N(0, I) prior, so it is compact and continuous |
| Generation | no principled way to sample | sample `z ~ N(0, I)` and decode |

Architecture (identical for both):

- **Encoder:** Conv(1→32, 4×4, stride 2) → BN → LeakyReLU → Conv(32→64, 4×4, stride 2) → BN → LeakyReLU → Linear(3136→256)
- **Latent head:** AE: Linear(256→d); VAE: Linear(256→d) for μ and Linear(256→d) for log σ²
- **Decoder:** Linear(d→256) → Linear(256→3136) → ConvTranspose(64→32) → BN → LeakyReLU → ConvTranspose(32→1) → logits (sigmoid gives pixels)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Usage

```bash
python main.py run                                    # train AE and VAE, then compare
python main.py train --model vae --epochs 30 --beta 1
python main.py compare                                # only evaluate saved checkpoints
python main.py run --latent-dim 2 --output-dir outputs/latent_2d
python -m pytest tests
```

Main options: `--epochs` (20), `--latent-dim` (16), `--batch-size` (128),
`--lr` (1e-3), `--beta` (1.0), `--device auto|cuda|cpu`, `--seed` (42).

MNIST downloads automatically to `data/`. The 60 000 training images are split
into 55 000 for training and 5 000 for validation; the 10 000 test images are
used only for the final comparison. The weights with the best validation loss
are kept.

## Outputs

`outputs/metrics/`: `comparison.json` (test metrics), `ae_history.json`,
`vae_history.json`, `config.json`.

`outputs/figures/`:

- `loss_curves.png`: training and validation loss, reconstruction term and KL term
- `reconstructions.png`: the same test digits reconstructed by each model
- `generated_samples.png`: 64 images decoded from `z ~ N(0, I)`
- `latent_space.png`: test-set codes coloured by digit (PCA to 2-D when d > 2)
- `interpolations.png`: decoding a straight line between two digits' codes
- `latent_manifold.png`: decoded grid over the latent plane (only when d = 2)

## Results

20 epochs, batch 128, Adam (lr 1e-3), β = 1, seed 42, trained on CPU.
Metrics are on the 10 000 test images. BCE and KL are per image (summed over
pixels / latent dimensions), MSE is per pixel, and reconstructions use the
deterministic code (z for the AE, μ for the VAE).

| Latent dim | Model | BCE ↓ | MSE ↓ | PSNR (dB) ↑ | KL | mean \|E[z]\| | mean std(z) |
|---|---|---|---|---|---|---|---|
| 16 | Autoencoder | **62.50** | **0.0062** | **23.08** | – | 2.55 | 9.97 |
| 16 | VAE | 66.49 | 0.0076 | 21.94 | 23.11 | 0.10 | 0.95 |
| 2 | Autoencoder | **136.85** | **0.0379** | **14.93** | – | 1.15 | 3.47 |
| 2 | VAE | 138.23 | 0.0385 | 14.84 | 6.34 | 0.10 | 1.04 |

Figures: `outputs/figures/` (d = 16) and `outputs/latent_2d/figures/` (d = 2).

### Observations

1. **Reconstruction.** The AE reconstructs slightly better (lower BCE, about
   1.1 dB higher PSNR at d = 16). It optimises reconstruction alone, while
   the VAE trades some reconstruction accuracy for the KL term. VAE
   reconstructions look a little blurrier. With d = 2 both models lose a lot of
   detail and the gap almost disappears, because the bottleneck dominates.
2. **Generation.** This is where the models differ most (`generated_samples.png`).
   Decoding `z ~ N(0, I)` with the VAE gives recognisable, varied digits. The
   AE produces almost black images: its codes have a spread of about 10 per
   dimension and are not centred at 0, so N(0, I) falls in a region the
   decoder never saw during training. The latent statistics in the table show
   this directly: the VAE's aggregate codes have mean ≈ 0 and std ≈ 1, matching
   the prior.
3. **Latent space structure.** The AE's latent space (`latent_space.png`) is
   wide and irregular, with clusters of different scale and gaps between them.
   The VAE's space is compact and centred, with classes laid out continuously
   next to each other. The 2-D grid (`latent_manifold.png`) shows it: every
   point of the VAE grid decodes to a plausible digit with smooth transitions,
   while the AE grid only covers part of its codes and contains blurry,
   ambiguous shapes in the regions between clusters.
4. **Interpolation.** Both models morph one digit into another. The VAE's
   intermediate images stay closer to real digit shapes, because the KL term
   makes the space between encoded points meaningful.
5. **Training dynamics** (`loss_curves.png`). The AE's loss drops fastest.
   The VAE's KL term settles around 23–24 nats at d = 16 (about 6 at d = 2),
   showing how much information per image the encoder puts into the latent
   code.

### Conclusion

The classic autoencoder is the better **compressor**: for the same bottleneck
it gives sharper, more accurate reconstructions. It learns no distribution over
its latent space, however, so it cannot be used to **generate** new data. The
VAE gives up a little reconstruction quality to regularise the latent space
toward N(0, I). That makes the space continuous and complete, so it can
generate new images by sampling, interpolate smoothly, and produce a
meaningful latent manifold. The β parameter (`--beta`) controls this
trade-off: a lower β gives sharper reconstructions, a higher β gives a
smoother latent space.

## Note on the GPU

On this machine `torch.cuda.is_available()` is True but kernels fail with
`cudaErrorDevicesUnavailable`, because NVIDIA driver 532 (CUDA 12.1) is older
than the CUDA 12.6 build of PyTorch. `autoencoders/device.py` detects this and
falls back to the CPU automatically (about 12 s per epoch). Updating the
NVIDIA driver will enable the GPU with no code changes.
