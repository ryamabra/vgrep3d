# Reproducibility

vgrep3d combines stochastic model training, external model weights, COLMAP, and GPU kernels. Record the following with every experiment so results can be compared honestly.

## Run record

- Git commit (`git rev-parse HEAD`)
- Scene name and image count
- Capture method and camera/view coverage
- COLMAP version and reconstruction settings
- gsplat, PyTorch, CUDA, Transformers, and Segment Anything versions
- SigLIP and SAM model/checkpoint identifiers
- Gaussian checkpoint path and full Gaussian count
- `max_gaussians`, latent dimension, epochs, views per epoch, learning rate, and random seed
- Query prompt, threshold, minimum hit count, and box percentiles
- Hardware and approximate wall-clock time

The checkpoint loader uses a fixed subsampling seed (`1234`). Changing that seed, or changing `max_gaussians`, changes the identity of every retained row and invalidates previously trained latent files.

## Recommended environment capture

```bash
git rev-parse HEAD
python --version
python -m pip freeze > environment.txt
nvidia-smi > gpu.txt
colmap --version
```

Keep environment captures beside experiment outputs, not in the repository. Avoid comparing support counts between runs unless the scene checkpoint, subsampling cap, prompt, and threshold all match.

Copy [`configs/experiment.example.json`](../configs/experiment.example.json) into an experiment output directory and update it with the actual values used. The file is a run-record template; current Modal commands do not consume it automatically.

## Determinism limits

Set NumPy and PyTorch seeds before training when exact sampling order matters. CUDA rasterization and some reduction kernels may still be nondeterministic. Treat small numeric differences as expected and report aggregate metrics across repeated runs for method comparisons.
