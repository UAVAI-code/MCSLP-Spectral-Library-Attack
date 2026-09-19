# MCSLP spectral-library poisoning attack

Material-Response Consensus Spectral Library Poisoning (MCSLP), implemented with
MOEA/D. It modifies a small number of library spectra using smooth cosine-basis
perturbations, enforces reflectance, amplitude, spectral-angle, and roughness
constraints, scores material-response changes across proxy unmixers, and performs
complete-image reevaluation before selecting the final candidate.

The package intentionally contains **algorithm code only**. It does not include
real or synthetic datasets, dataset loaders, other unmixing algorithms, baseline
implementations, ablation code, NSGA-II/random-search variants, figures, or
experiment scripts.

## Interface

The caller supplies an `AttackScene` with an `H x W x B` observation cube, a
`B x M` spectral library, and public library groups. Proxy unmixers are supplied
through a callback with this signature:

```python
from mcslp_attack import ProxyEstimate

def proxy_runner(scene, method_name, proxy_config):
    # Return abundance with shape M x N and selected zero-based atom indices.
    return ProxyEstimate(abundance, selected_support)
```

Run the released attack as follows:

```python
from mcslp_attack import ProxyAttackEvaluator, SpectralAttackConfig, run_moead_attack

config = SpectralAttackConfig(attack_atoms=3, seed=0)
evaluator = ProxyAttackEvaluator(
    scene, ["proxy_a", "proxy_b", "proxy_c"], proxy_runner, config
)
result = run_moead_attack(evaluator, config)
poisoned_library = result.attacked_library
```

The attack evaluator receives only the observation, library, public group
metadata, and proxy outputs. Ground-truth abundances and supports are not part
of the released input interface.

## Requirements

- Python 3.10+
- NumPy

Install the only runtime dependency with:

```bash
pip install -r requirements.txt
```

The implementation version is `v4.2-direction-gated-global`.
