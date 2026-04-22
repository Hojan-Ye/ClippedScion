# Modded NanoGPT

This code builds on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/).

## Setup

```bash
pip install -r requirements.txt
pip install -r data/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade
python data/cached_fineweb10B.py 8 # downloads only the first 800M training tokens to save time
```

## Run

- 124M

```bash
# Unconstrained ClippedScion
torchrun --standalone --nproc_per_node=4 train_gpt_clippedscion.py --warmdown-iters 1
# or
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt_clippedscion.py --warmdown-iters 1 --device-batch-size 32

# ClippedScion
torchrun --standalone --nproc_per_node=4 train_gpt_clippedscion.py --warmdown-iters 1 --unconstrained False
# or
torchrun --standalone --nproc_per_node=1 train_gpt_clippedscion.py --warmdown-iters 1 --device-batch-size 32 --unconstrained
```

- 1B

```bash
# Unconstrained ClippedScion
torchrun --standalone --nproc_per_node=4 train_gpt_clippedscion.py --warmdown-iters 1 --n-embd 2560 --n-head 20 --device-batch-size 16 --rho 6000
```

Notes: 

- When changing `n_embd`, remember to change `n_head` accordingly to `n_embd // 128` to maintain head dimension of 128.

