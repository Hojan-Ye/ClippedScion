# ClippedScion

Code accompanying the paper [Generalized Gradient Norm Clipping
\& Non-Euclidean $(L_0,L_1)$-Smoothness](https://arxiv.org/pdf/2506.01913).

This paper is a following work of [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/pdf/2502.07529) and is based on the [Scion codebase](https://github.com/LIONS-EPFL/scion).

## Repository structure

- [`clippedscion.py`](clippedscion.py): Contains the `UnconstrainedClippedScion` and `ClippedScion` reference implementation along with various norm choices. 
    - [Algorithm 3](https://arxiv.org/pdf/2506.01913) corresponds to `UnconstrainedClippedScion`.
    - [Algorithm 4 (Variant 2)](https://arxiv.org/pdf/2506.01913) corresponds to `ClippedScion`. For simplicity, we control $\min\{\rho, \sum_{l=1}^D \braket{d^k_l,v^k_l}\}$ in practice.
- [`examples/`](examples/): Example usage containing airbench, nanoGPT, and DeiT experiments with and without weight sharing.

## Notes

The `ClippedScion` optimizer comes with a couple of hyperparameters:

- `momentum`: The parameter is `1-usual_momentum` of e.g. the PyTorch implementation of SGD with momentum. 
    A good default is 0.1. 
    Higher values seem to work better (e.g. 0.5) for short training runs with low noise as also supported by theory.
- `scale`: Controls the per-layer constraint radius factor. 
    The layerwise radius can be tuned on a small proxy model similarly to the input and output scaling factor of µP.
- `lr`: The learning rate can similarly be tuned on a small proxy model (corresponds to γ in the paper).
- `unconstrained`: When set to `False` the constrained variant of the ClippedScion is used, which guarantees the iterates to stay bounded.
- `rho`: Clipping threshold controls $\sum_{l=1}^D \braket{d^k_l,v^k_l}$ in [Algorithm 3 & 4](https://arxiv.org/pdf/2506.01913).

Architectural changes:

- Scale activation functions (ReLU, GELU) [by √2](https://github.com/LIONS-EPFL/scion/blob/main/examples/shallow-nanogpt/model.py#L104) to maintain the input variance.


## Examples

For runnable examples see [`examples/`](examples/).
Below are some pseudocode configurations for different architectures and domains (see [Appendix C](https://arxiv.org/pdf/2506.01913) for exact parameter choices):


- nanoGPT with weight sharing (see [`examples/modded-nanogpt`](examples/modded-nanogpt)):

    ```python
    radius = 50.0
    threshold = 600
    optim_groups = [{
        'params': model.transformer.h.parameters(),
        'norm': 'Spectral',
        'norm_kwargs': {},
        'scale': radius,
    }, {
        'params': model.lm_head.parameters(),
        'norm': 'Sign',
        'norm_kwargs': {},
        'scale': radius*60.0,
    }]
    optimizer = UnconstrainedClippedScion(optim_groups, lr=2**-12, momentum=0.1, rho=600)
    ```

- CNN (see [`examples/airbench`](examples/airbench) for further details):

    ```python
    radius = 8.0
    threshold = 1600
    optim_groups = [{
        'params': remaining_parameters,
        'norm': 'Auto', # Picks layerwise norm based on the parameter shape
        'norm_kwargs': {},
        'scale': radius,
    }, {
        'params': output_layer,
        'norm': 'Sign',
        'norm_kwargs': {'normalized': True},
        'scale': radius*16,
    }]
    optimizer = UnconstrainedClippedScion(optim_groups, lr=2**-4, momentum=0.5, rho=1600)
    ```
- DeiT
    ```python
    radius = 25
    threshold = 8000
    optim_groups = [{
        'params': other_params,
        'norm': 'Auto',
        'norm_kwargs': {},
        'scale': radius,
    },{
        'params': head_weights,
        'norm': 'Sign',
        'norm_kwargs': {},
        'scale': radius*20,
    },{
        'params': [pos_embed_param, cls_token_param],
        'norm': 'BiasRMS',
        'norm_kwargs': {},
        'scale': radius,
    }]
    optimizer = UnconstrainedClippedScion(optim_groups, lr=8e-5, momentum=0.1, rho=8000)
    ```

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@article{pethick2025generalized,
  title={Generalized Gradient Norm Clipping \& Non-Euclidean $(L\_0, L\_1) $-Smoothness},
  author={Pethick, Thomas and Xie, Wanyun and Erdogan, Mete and Antonakopoulos, Kimon and Silveti-Falls, Tony and Cevher, Volkan},
  journal={arXiv preprint arXiv:2506.01913},
  year={2025}
}
```
