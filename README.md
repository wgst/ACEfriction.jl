<p align="right">
  <a href="https://acesuit.github.io/ACEfriction.jl/dev/">
    <img src="https://github.com/ACEsuit/ACEfriction.jl/blob/main/docs/src/assets/logo.png" alt="ACEfriction.jl logo"
         title="ACEfriction.jl" align="right" height="60"/>
  </a>
</p>

# ACEfriction.jl

| **Documentation**                                     | **Build Status**                                |  **License**                     |
|:------------------------------------------------------|:----------------------------------------------- |:-------------------------------- |
| [![][docs-img]][docs-url] [![][ddocs-img]][ddocs-url] | [![][ci-img]][ci-url] [![][ccov-img]][ccov-url] | [![][license-img]][license-url]  |

[ddocs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[ddocs-url]: https://acesuit.github.io/ACEfriction.jl/dev/

[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-url]: https://acesuit.github.io/ACEfriction.jl/dev/

[ci-img]: https://github.com/ACEsuit/ACEfriction.jl/actions/workflows/Tests.yml/badge.svg
[ci-url]: https://github.com/ACEsuit/ACEfriction.jl/actions/workflows/CI.yml

[ccov-img]: https://codecov.io/gh/ACEsuit/ACEfriction.jl/branch/main/graph/badge.svg
[ccov-url]: https://codecov.io/gh/ACEsuit/ACEfriction.jl

[license-img]: https://img.shields.io/github/license/ACEsuit/ACEfriction.jl
[license-url]: https://github.com/ACEsuit/ACEfriction.jl/blob/main/LICENSE

## About ACEfriction.jl

ACEfriction.jl facilitates simulation and machine learning of configuration-dependent friction tensor models from data. In more general terms, it allows efficient representation, learning, and evaluation of $E(3)$-equivariant symmetric positive semi-definite matrix-valued functions on 3D-point clouds, i.e., $E(3)$-equivariant functions of the form
```math
{\bf \Gamma}({\bf r}_1, \dots, {\bf r}_N, {z_1},\dots,{z_N}) \in \mathcal{SPSD}_{3N},
```
where $`{\bf r}_i \in \mathbb{R}^3,\; (i=1,\dots,N)`$, are the positions of points in the point cloud, the $`z_i`$s are some discrete features (e.g., chemical element types) and $\mathcal{SPSD}_{3N} \subset \mathbb{R}^{3N \times 3N}$ is the set of $3N\times 3N$-dimensional positive semi-definite matrices.

The underlying representation is based on an equivariant Atomic Cluster Expansion and, as such, size-transferrable, i.e., models can be trained and evaluated on a 3D-point cloud comprised of an arbitrary number of particles $N$.    

## Documentation

For details, please refer to the [Documentation](https://acesuit.github.io/ACEfriction.jl/dev/), which includes a function manual and [Workflow Examples](https://acesuit.github.io/ACEfriction.jl/dev/fitting-eft/) of fitting an Electronic Friction Tensor as well as a momentum-conserving friction tensor model as commonly employed in Dissipative Particle Dynamics.  

## Installation
To install ACEfriction.jl run the following code in a Julia-REPL:
```julia-repl
] registry add https://github.com/ACEsuit/ACEregistry
] add ACEfriction
```
More detailed instructions can be found in the [Installation Guide](https://acesuit.github.io/ACEfriction.jl/dev/installation/) of the Documentation. 

## Reference
If you use this code, please cite our paper:
```bibtex
@article{sachs2024equivariant,
  title={Equivariant Representation of Configuration-Dependent Friction Tensors in Langevin Heatbaths},
  author={Sachs, Matthias and Stark, Wojciech G and Maurer, Reinhard J and Ortner, Christoph},
  journal={arXiv preprint arXiv:2407.13935},
  year={2024}
}
```

## License

ACEfriction.jl is published and distributed under the [MIT License](LICENSE).
