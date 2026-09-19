"""Material-response consensus spectral-library poisoning with MOEA/D.

This release contains only the paper's complete MCSLP attack algorithm.  Proxy
unmixers are supplied by the caller through ``proxy_runner``; no dataset loader,
real data, or unmixing implementation is bundled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Iterable

import numpy as np

EPS = 1e-10
ATTACK_MODEL_VERSION = "v4.2-direction-gated-global"


@dataclass
class AttackScene:
    """Minimal attack input; truth fields are intentionally absent."""
    name: str
    cube: np.ndarray                 # H x W x B
    library: np.ndarray              # B x M
    library_groups: tuple[np.ndarray, ...] | None = None
    wavelengths: np.ndarray | None = None
    material_names: tuple[str, ...] = ()
    evidence: str = "caller supplied"

    @property
    def observation(self) -> np.ndarray:
        return np.moveaxis(np.asarray(self.cube), -1, 0).reshape(self.cube.shape[-1], -1)


@dataclass
class ProxyEstimate:
    """Output required from one caller-supplied proxy unmixing method."""
    abundance: np.ndarray             # M x N, in the input library coordinates
    selected_support: np.ndarray      # zero-based atom indices


def normalize_abundance(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return values / np.maximum(values.sum(axis=0, keepdims=True), EPS)


def grouped_abundance(abundance: np.ndarray, groups: tuple[np.ndarray, ...]) -> np.ndarray:
    values = np.asarray(abundance, dtype=np.float64)
    return np.stack([values[np.asarray(group, dtype=np.int64)].sum(axis=0) for group in groups], axis=0)


def attack_model_version(config: "SpectralAttackConfig") -> str:
    return ATTACK_MODEL_VERSION

@dataclass
class SpectralAttackConfig:
    attack_atoms: int = 2
    basis_dimension: int = 6
    linf_cap: float = 0.02
    sam_cap_radians: float = 0.02
    roughness_cap: float = 0.002
    population_size: int = 8
    generations: int = 5
    neighbor_size: int = 3
    differential_weight: float = 0.6
    crossover_rate: float = 0.8
    atom_swap_probability: float = 0.65
    coefficient_mutation_probability: float = 0.25
    coefficient_mutation_scale: float = 0.20
    transfer_quantile: float = 0.0
    objective_mode: str = "consensus_v4_group_directional"
    group_divergence_weight: float = 0.50
    global_group_shift_weight: float = 0.50
    direction_coverage_power: float = 1.00
    direction_confidence_quantile: float = 0.50
    active_weight_power: float = 2.0
    inactive_shift_penalty: float = 0.10
    physical_mismatch_weight: float = 0.25
    residual_growth_penalty: float = 0.10
    support_exploration_probability: float = 0.15
    moead_min_attack_weight: float = 0.50
    random_immigrant_probability: float = 0.25
    coefficient_restart_probability: float = 0.20
    multi_atom_swap_probability: float = 0.30
    support_learning_rate: float = 0.35
    stagnation_generations: int = 3
    stagnation_immigrant_probability: float = 0.50
    stagnation_coefficient_restart_probability: float = 0.40
    stagnation_multi_atom_swap_probability: float = 0.60
    max_neighbor_replacements: int = 2
    max_support_copies: int = 2
    report_sam_threshold: float | None = None
    seed: int = 0



@dataclass
class AttackCandidate:
    support: np.ndarray
    coefficients: np.ndarray
    objectives: np.ndarray | None = None
    measurements: dict[str, object] = field(default_factory=dict)

    def clone(self) -> "AttackCandidate":
        return AttackCandidate(
            support=self.support.copy(),
            coefficients=self.coefficients.copy(),
            objectives=None if self.objectives is None else self.objectives.copy(),
            measurements=dict(self.measurements),
        )


@dataclass
class SpectralAttackResult:
    selected: AttackCandidate
    attacked_library: np.ndarray
    delta: np.ndarray
    pareto_archive: list[AttackCandidate]
    search_candidates: list[AttackCandidate]
    candidate_queries: int
    solver_queries: int
    clean_proxy_diagnostics: dict[str, object]
    history: list[dict[str, float | int]]


def low_frequency_cosine_basis(bands: int, dimension: int) -> np.ndarray:
    """Return smooth cosine atoms with unit peak magnitude."""
    if bands < 3:
        raise ValueError("At least three spectral bands are required")
    dimension = min(max(1, int(dimension)), bands)
    locations = np.arange(bands, dtype=np.float64) + 0.5
    frequencies = np.arange(dimension, dtype=np.float64)
    basis = np.cos(np.pi * locations[:, None] * frequencies[None, :] / bands)
    basis /= np.maximum(np.max(np.abs(basis), axis=0, keepdims=True), EPS)
    return basis


def spectral_angle(first: np.ndarray, second: np.ndarray) -> float:
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    cosine = float(
        np.dot(first64, second64)
        / max(float(np.linalg.norm(first64) * np.linalg.norm(second64)), EPS)
    )
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def spectral_roughness(delta: np.ndarray) -> float:
    delta64 = np.asarray(delta, dtype=np.float64)
    if delta64.size < 3:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(delta64, n=2) ** 2)))


def _bounded_atom(
    clean: np.ndarray,
    raw_delta: np.ndarray,
    config: SpectralAttackConfig,
) -> np.ndarray:
    clean64 = np.asarray(clean, dtype=np.float64)
    delta = np.asarray(raw_delta, dtype=np.float64).copy()
    maximum = float(np.max(np.abs(delta)))
    if maximum > config.linf_cap:
        delta *= config.linf_cap / maximum
    roughness = spectral_roughness(delta)
    if config.roughness_cap > 0.0 and roughness > config.roughness_cap:
        delta *= config.roughness_cap / roughness

    def atom_at(scale: float) -> np.ndarray:
        # Evaluate the exact representation saved and audited downstream.  In
        # particular, reflectance clipping can add a sharp corner to the final
        # perturbation even when the pre-clipping delta satisfies roughness.
        return np.clip(clean64 + scale * delta, 0.0, 1.0).astype(np.float32)

    scale = 1.0
    for _ in range(32):
        attacked = atom_at(scale)
        actual_delta = attacked.astype(np.float64) - clean64
        final_linf = float(np.max(np.abs(actual_delta)))
        final_roughness = spectral_roughness(actual_delta)
        final_sam = spectral_angle(clean64, attacked)
        linf_ok = final_linf <= config.linf_cap
        roughness_ok = (
            config.roughness_cap <= 0.0
            or final_roughness <= config.roughness_cap
        )
        sam_ok = final_sam <= config.sam_cap_radians
        if linf_ok and roughness_ok and sam_ok:
            return attacked

        shrink_factors = [1.0]
        if not linf_ok:
            shrink_factors.append(config.linf_cap / max(final_linf, EPS))
        if not roughness_ok:
            shrink_factors.append(
                config.roughness_cap / max(final_roughness, EPS)
            )
        if not sam_ok:
            shrink_factors.append(
                config.sam_cap_radians / max(final_sam, EPS)
            )
        # A small interior margin absorbs float32 rounding in the saved library.
        scale *= max(0.0, min(shrink_factors)) * (1.0 - 1e-6)

    # Zero perturbation is always feasible and is preferable to emitting an
    # invalid sample if numerical corner cases prevent convergence above.
    return np.asarray(clean, dtype=np.float32).copy()


def construct_attacked_library(
    clean_library: np.ndarray,
    candidate: AttackCandidate,
    basis: np.ndarray,
    config: SpectralAttackConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    clean = np.asarray(clean_library, dtype=np.float32)
    attacked = clean.copy()
    if len(candidate.support) != config.attack_atoms:
        raise ValueError("Candidate support does not match attack_atoms")
    if candidate.coefficients.shape != (config.attack_atoms, basis.shape[1]):
        raise ValueError("Candidate coefficient shape does not match the attack basis")
    support = np.asarray(candidate.support, dtype=np.int64)
    if len(np.unique(support)) != config.attack_atoms:
        raise ValueError("Candidate support contains duplicate atoms")
    if np.any(support < 0) or np.any(support >= clean.shape[1]):
        raise ValueError("Candidate support contains an out-of-range atom index")
    clean_support = clean[:, support]
    support_is_attackable = np.all(
        np.isfinite(clean_support)
        & (clean_support >= 0.0)
        & (clean_support <= 1.0),
        axis=0,
    )
    if not np.all(support_is_attackable):
        invalid = support[~support_is_attackable]
        raise ValueError(
            "Candidate support contains a pre-existing nonphysical atom: "
            + ", ".join(str(int(value)) for value in invalid)
        )
    for row, atom_index in enumerate(candidate.support):
        raw_delta = basis @ candidate.coefficients[row]
        attacked[:, int(atom_index)] = _bounded_atom(clean[:, int(atom_index)], raw_delta, config)
    delta = attacked - clean
    angles = [spectral_angle(clean[:, int(j)], attacked[:, int(j)]) for j in candidate.support]
    linf_values = [float(np.max(np.abs(delta[:, int(j)]))) for j in candidate.support]
    roughness_values = [spectral_roughness(delta[:, int(j)]) for j in candidate.support]
    attacked_support = attacked[:, support]
    clean_invalid = ~np.isfinite(clean) | (clean < 0.0) | (clean > 1.0)
    attacked_invalid = (
        ~np.isfinite(attacked) | (attacked < 0.0) | (attacked > 1.0)
    )
    new_invalid = attacked_invalid & ~clean_invalid
    constraints = {
        "support_zero_based": [int(value) for value in candidate.support],
        "mean_sam_radians": float(np.mean(angles)),
        "max_sam_radians": float(np.max(angles)),
        "max_linf": float(np.max(linf_values)),
        "max_roughness": float(np.max(roughness_values)),
        "reflectance_min": float(attacked_support.min()),
        "reflectance_max": float(attacked_support.max()),
        "attacked_support_reflectance_min": float(attacked_support.min()),
        "attacked_support_reflectance_max": float(attacked_support.max()),
        "clean_library_reflectance_min": float(np.nanmin(clean)),
        "clean_library_reflectance_max": float(np.nanmax(clean)),
        "clean_library_out_of_bounds_entries": int(np.sum(clean_invalid)),
        "attacked_library_out_of_bounds_entries": int(np.sum(attacked_invalid)),
        "new_out_of_bounds_entries": int(np.sum(new_invalid)),
    }
    constraints["feasible"] = bool(
        constraints["max_sam_radians"] <= config.sam_cap_radians + 1e-7
        and constraints["max_linf"] <= config.linf_cap + 1e-7
        and constraints["max_roughness"] <= config.roughness_cap + 1e-7
        and constraints["reflectance_min"] >= -1e-7
        and constraints["reflectance_max"] <= 1.0 + 1e-7
        and constraints["new_out_of_bounds_entries"] == 0
    )
    return attacked, delta, constraints


def scene_with_library(scene: AttackScene, library: np.ndarray) -> AttackScene:
    return AttackScene(
        name=scene.name,
        cube=scene.cube,
        library=np.asarray(library, dtype=np.float32),
        wavelengths=scene.wavelengths,
        # Library grouping is public dictionary metadata, not pixelwise truth.
        # V4 needs it to optimize the same material-level representation used
        # by the released real-scene evaluation protocol.
        library_groups=scene.library_groups,
        material_names=scene.material_names,
        evidence=scene.evidence,
    )


def pixel_subset_scene(scene: AttackScene, pixel_indices: np.ndarray) -> AttackScene:
    """Make a 1 x N scene without carrying truth into the attack evaluator."""
    observation = scene.observation[:, np.asarray(pixel_indices, dtype=np.int64)]
    cube = observation.T[None, :, :].astype(np.float32)
    return AttackScene(
        name=f"{scene.name}_attack_subset",
        cube=cube,
        library=scene.library,
        wavelengths=scene.wavelengths,
        library_groups=scene.library_groups,
        material_names=scene.material_names,
        evidence="truth-stripped attack search subset",
    )


class ProxyAttackEvaluator:
    """Evaluate library candidates using only proxy-unmixer outputs."""

    def __init__(
        self,
        scene: AttackScene,
        methods: Iterable[str],
        proxy_runner: Callable[[AttackScene, str, object | None], ProxyEstimate],
        attack_config: SpectralAttackConfig,
        proxy_config: object | None = None,
    ) -> None:
        self.scene = scene_with_library(scene, scene.library)
        self.methods = tuple(methods)
        if not self.methods:
            raise ValueError("At least one proxy method is required")
        self.proxy_runner = proxy_runner
        self.proxy_config = proxy_config
        self.attack_config = attack_config
        self.library_groups = self.scene.library_groups
        if (
            attack_config.objective_mode == "consensus_v4_group_directional"
            and self.library_groups is None
        ):
            raise ValueError(
                "consensus_v4_group_directional requires public library_groups; "
                "use consensus_v3 for ungrouped libraries"
            )
        self.attackable_mask = np.all(
            np.isfinite(scene.library)
            & (scene.library >= 0.0)
            & (scene.library <= 1.0),
            axis=0,
        )
        if int(np.sum(self.attackable_mask)) < attack_config.attack_atoms:
            raise ValueError(
                "The clean library does not contain enough physically valid atoms "
                f"for q={attack_config.attack_atoms}"
            )
        self.basis = low_frequency_cosine_basis(scene.library.shape[0], attack_config.basis_dimension)
        self.clean_outputs: dict[str, np.ndarray] = {}
        self.clean_group_outputs: dict[str, np.ndarray] = {}
        self.clean_supports: dict[str, list[int]] = {}
        self.clean_relative_residuals: dict[str, float] = {}
        observation = self.scene.observation.astype(np.float64)
        dictionary = self.scene.library.astype(np.float64)
        observation_norm = max(float(np.linalg.norm(observation)), EPS)
        normalized_energies: list[np.ndarray] = []
        support_votes = np.zeros(self.scene.library.shape[1], dtype=np.float64)
        for method in self.methods:
            estimate = self.proxy_runner(self.scene, method, self.proxy_config)
            abundance = np.asarray(estimate.abundance, dtype=np.float32)
            self.clean_outputs[method] = abundance
            if self.library_groups is not None:
                self.clean_group_outputs[method] = grouped_abundance(
                    abundance, self.library_groups
                )
            self.clean_supports[method] = [int(value) for value in estimate.selected_support]
            energy = np.linalg.norm(abundance.astype(np.float64), axis=1)
            energy /= max(float(energy.max()), EPS)
            normalized_energies.append(energy)
            support_votes[np.asarray(estimate.selected_support, dtype=np.int64)] += 1.0
            self.clean_relative_residuals[method] = float(
                np.linalg.norm(observation - dictionary @ abundance.astype(np.float64))
                / observation_norm
            )

        energy_consensus = np.median(np.stack(normalized_energies), axis=0)
        vote_fraction = support_votes / len(self.methods)
        consensus = energy_consensus * (0.25 + 0.75 * vote_fraction)
        consensus = np.where(self.attackable_mask, consensus, 0.0)
        if float(consensus.max()) <= EPS:
            consensus = np.ones_like(consensus)
        consensus /= max(float(consensus.max()), EPS)
        self.active_weights = np.power(
            np.clip(consensus, 0.0, 1.0), attack_config.active_weight_power
        )
        self.group_consensus: np.ndarray | None = None
        self.direction_source: np.ndarray | None = None
        self.direction_target: np.ndarray | None = None
        self.direction_pixel_weights: np.ndarray | None = None
        if self.clean_group_outputs:
            stacked_groups = np.stack(list(self.clean_group_outputs.values()), axis=0)
            self.group_consensus = normalize_abundance(np.median(stacked_groups, axis=0))
            if self.group_consensus.shape[0] < 2:
                raise ValueError("The directional group objective requires at least two groups")
            order = np.argsort(self.group_consensus, axis=0)
            self.direction_source = order[-1].astype(np.int64)
            self.direction_target = order[-2].astype(np.int64)
            source_values = np.take_along_axis(
                self.group_consensus, self.direction_source[None, :], axis=0
            )[0]
            target_values = np.take_along_axis(
                self.group_consensus, self.direction_target[None, :], axis=0
            )[0]
            confidence = np.maximum(source_values - target_values, 0.0)
            threshold = float(
                np.quantile(
                    confidence,
                    np.clip(attack_config.direction_confidence_quantile, 0.0, 1.0),
                )
            )
            agreement = np.mean(
                np.stack(
                    [
                        np.argmax(values, axis=0) == self.direction_source
                        for values in self.clean_group_outputs.values()
                    ],
                    axis=0,
                ),
                axis=0,
            )
            weights = confidence * agreement * (confidence >= threshold)
            if float(weights.sum()) <= EPS:
                weights = np.maximum(confidence, EPS)
            self.direction_pixel_weights = weights / max(float(weights.mean()), EPS)
        # V4.2 uses material groups in the attack score.  Support proposals
        # deliberately keep the same atom-activity consensus for every
        self._importance = consensus
        self._observation = observation
        self._observation_norm = observation_norm
        self._dictionary = dictionary
        self.cache: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
        self.candidate_queries = 0
        self.solver_queries = len(self.methods)

    def importance(self) -> np.ndarray:
        return self._importance.copy()

    def evaluate(self, candidate: AttackCandidate) -> AttackCandidate:
        attacked, _, constraints = construct_attacked_library(
            self.scene.library, candidate, self.basis, self.attack_config
        )
        digest = hashlib.sha256(np.round(attacked, decimals=7).tobytes()).hexdigest()
        cached = self.cache.get(digest)
        if cached is not None:
            candidate.objectives = cached[0].copy()
            candidate.measurements = dict(cached[1])
            return candidate

        shifts: dict[str, float] = {}
        active_shifts: dict[str, float] = {}
        inactive_shifts: dict[str, float] = {}
        physical_mismatch: dict[str, float] = {}
        attacked_residual_growth: dict[str, float] = {}
        proxy_scores_v2: dict[str, float] = {}
        proxy_scores_v3: dict[str, float] = {}
        group_divergences: dict[str, float] = {}
        global_group_divergences: dict[str, float] = {}
        global_group_shifts: dict[str, float] = {}
        relative_source_drops: dict[str, float] = {}
        relative_target_gains: dict[str, float] = {}
        directional_transfers: dict[str, float] = {}
        direction_coverages: dict[str, float] = {}
        direction_consistent: dict[str, bool] = {}
        proxy_scores_v4_ungated: dict[str, float] = {}
        proxy_scores_v4: dict[str, float] = {}
        proxy_scores: dict[str, float] = {}
        attacked_scene = scene_with_library(self.scene, attacked)
        for method in self.methods:
            estimate = self.proxy_runner(attacked_scene, method, self.proxy_config)
            clean = self.clean_outputs[method].astype(np.float64)
            changed = estimate.abundance.astype(np.float64)
            abundance_delta = changed - clean
            shifts[method] = float(
                np.linalg.norm(abundance_delta) / (np.linalg.norm(clean) + EPS)
            )
            active_scale = np.sqrt(self.active_weights)[:, None]
            inactive_scale = np.sqrt(
                np.clip(1.0 - self.active_weights, 0.0, 1.0)
            )[:, None]
            active_denominator = np.linalg.norm(active_scale * clean) + EPS
            active_shifts[method] = float(
                np.linalg.norm(active_scale * abundance_delta) / active_denominator
            )
            inactive_shifts[method] = float(
                np.linalg.norm(inactive_scale * abundance_delta)
                / (np.linalg.norm(clean) + EPS)
            )
            clean_residual = max(self.clean_relative_residuals[method], EPS)
            attacked_clean_dictionary_residual = float(
                np.linalg.norm(self._observation - self._dictionary @ changed)
                / self._observation_norm
            )
            physical_mismatch[method] = (
                attacked_clean_dictionary_residual - clean_residual
            ) / clean_residual
            attacked_dictionary_residual = float(
                np.linalg.norm(
                    self._observation - attacked.astype(np.float64) @ changed
                )
                / self._observation_norm
            )
            attacked_residual_growth[method] = (
                attacked_dictionary_residual - clean_residual
            ) / clean_residual
            proxy_scores_v2[method] = float(
                active_shifts[method]
                - self.attack_config.inactive_shift_penalty * inactive_shifts[method]
                + self.attack_config.physical_mismatch_weight * physical_mismatch[method]
            )
            proxy_scores_v3[method] = float(
                active_shifts[method]
                - self.attack_config.inactive_shift_penalty * inactive_shifts[method]
                - self.attack_config.residual_growth_penalty
                * max(0.0, attacked_residual_growth[method])
            )
            if self.library_groups is not None:
                clean_group = self.clean_group_outputs[method]
                changed_group = grouped_abundance(changed, self.library_groups)
                midpoint = np.maximum(0.5 * (clean_group + changed_group), EPS)
                clean_safe = np.maximum(clean_group, EPS)
                changed_safe = np.maximum(changed_group, EPS)
                js_per_pixel = 0.5 * np.sum(
                    clean_safe * np.log(clean_safe / midpoint)
                    + changed_safe * np.log(changed_safe / midpoint),
                    axis=0,
                )
                weights = self.direction_pixel_weights
                assert weights is not None
                group_divergences[method] = float(
                    np.sum(weights * np.sqrt(np.maximum(js_per_pixel, 0.0)))
                    / (np.sum(weights) + EPS)
                )
                global_group_divergences[method] = float(
                    np.mean(np.sqrt(np.maximum(js_per_pixel, 0.0)))
                )
                global_group_shifts[method] = float(
                    np.linalg.norm(changed_group - clean_group)
                    / (np.linalg.norm(clean_group) + EPS)
                )
                assert self.direction_source is not None
                assert self.direction_target is not None
                clean_source = np.take_along_axis(
                    clean_group, self.direction_source[None, :], axis=0
                )[0]
                changed_source = np.take_along_axis(
                    changed_group, self.direction_source[None, :], axis=0
                )[0]
                clean_target = np.take_along_axis(
                    clean_group, self.direction_target[None, :], axis=0
                )[0]
                changed_target = np.take_along_axis(
                    changed_group, self.direction_target[None, :], axis=0
                )[0]
                relative_source_drops[method] = float(
                    np.sum(weights * (clean_source - changed_source))
                    / (np.sum(weights * clean_source) + EPS)
                )
                relative_target_gains[method] = float(
                    np.sum(weights * (changed_target - clean_target))
                    / (np.sum(weights * clean_target) + EPS)
                )
                directional_transfers[method] = float(
                    np.clip(
                        0.5
                        * (
                            relative_source_drops[method]
                            + relative_target_gains[method]
                        ),
                        -1.0,
                        1.0,
                    )
                )
                direction_coverages[method] = float(
                    np.sum(
                        weights
                        * (
                            (changed_source < clean_source)
                            & (changed_target > clean_target)
                        )
                    )
                    / (np.sum(weights) + EPS)
                )
                ungated_score = float(
                    self.attack_config.group_divergence_weight
                    * global_group_divergences[method]
                    + self.attack_config.global_group_shift_weight
                    * global_group_shifts[method]
                ) * float(
                    direction_coverages[method]
                    ** self.attack_config.direction_coverage_power
                ) - self.attack_config.residual_growth_penalty * max(
                    0.0, attacked_residual_growth[method]
                )
                proxy_scores_v4_ungated[method] = ungated_score
                direction_consistent[method] = bool(
                    directional_transfers[method] > 0.0
                    and direction_coverages[method] > 0.0
                )
                if direction_consistent[method]:
                    proxy_scores_v4[method] = ungated_score
                else:
                    # Direction is a semantic feasibility gate in V4.2.
                    proxy_scores_v4[method] = float(
                        -1.0
                        + min(0.0, directional_transfers[method])
                        - self.attack_config.residual_growth_penalty
                        * max(0.0, attacked_residual_growth[method])
                    )
            self.solver_queries += 1
        robust_shift = float(
            np.quantile(np.asarray(list(shifts.values()), dtype=np.float64), self.attack_config.transfer_quantile)
        )
        if self.attack_config.objective_mode != "consensus_v4_group_directional":
            raise ValueError("The released package implements consensus_v4_group_directional only")
        proxy_scores = proxy_scores_v4
        attack_score = float(
            np.quantile(
                np.asarray(list(proxy_scores.values()), dtype=np.float64),
                self.attack_config.transfer_quantile,
            )
        )
        stealth = float(constraints["mean_sam_radians"])
        objectives = np.asarray([-attack_score, stealth], dtype=np.float64)
        measurements: dict[str, object] = {
            **constraints,
            "robust_attack_score": attack_score,
            "robust_abundance_shift": robust_shift,
            "per_proxy_abundance_shift": shifts,
            "per_proxy_active_shift": active_shifts,
            "per_proxy_inactive_shift": inactive_shifts,
            "per_proxy_physical_mismatch": physical_mismatch,
            "per_proxy_attacked_residual_growth": attacked_residual_growth,
            "per_proxy_attack_score_v2": proxy_scores_v2,
            "per_proxy_attack_score_v3": proxy_scores_v3,
            "per_proxy_group_js_distance": group_divergences,
            "per_proxy_global_group_js_distance": global_group_divergences,
            "per_proxy_global_group_shift": global_group_shifts,
            "per_proxy_relative_source_drop": relative_source_drops,
            "per_proxy_relative_target_gain": relative_target_gains,
            "per_proxy_directional_transfer": directional_transfers,
            "per_proxy_direction_coverage": direction_coverages,
            "per_proxy_direction_consistent": direction_consistent,
            "per_proxy_attack_score_v4_ungated": proxy_scores_v4_ungated,
            "per_proxy_attack_score_v4": proxy_scores_v4,
            "per_proxy_attack_score": proxy_scores,
        }
        self.candidate_queries += 1
        self.cache[digest] = (objectives.copy(), dict(measurements))
        candidate.objectives = objectives
        candidate.measurements = measurements
        return candidate

    def diagnostics(self) -> dict[str, object]:
        ranked = np.argsort(-self._importance)
        return {
            "methods": list(self.methods),
            "clean_selected_support_zero_based": self.clean_supports,
            "clean_relative_residuals": self.clean_relative_residuals,
            "consensus_importance": self._importance.tolist(),
            "importance_mode": (
                "atom_activity_consensus_with_group_scoring"
                "atom_activity_consensus_with_group_scoring"
            ),
            "consensus_importance_rank_zero_based": [int(value) for value in ranked],
            "active_objective_weights": self.active_weights.tolist(),
            "attackable_atom_mask": self.attackable_mask.tolist(),
            "attackable_atoms_zero_based": [
                int(value) for value in np.flatnonzero(self.attackable_mask)
            ],
            "excluded_nonphysical_atoms_zero_based": [
                int(value) for value in np.flatnonzero(~self.attackable_mask)
            ],
            "search_scene_shape": list(self.scene.cube.shape),
            "library_groups_available_to_attack": self.library_groups is not None,
            "library_group_sizes": (
                []
                if self.library_groups is None
                else [int(len(group)) for group in self.library_groups]
            ),
            "direction_confident_pixels": (
                0
                if self.direction_pixel_weights is None
                else int(np.sum(self.direction_pixel_weights > 0.0))
            ),
            "ground_truth_available_to_attack": False,
        }


def _candidate_key(candidate: AttackCandidate) -> tuple[object, ...]:
    support_order = np.argsort(candidate.support)
    support = tuple(int(value) for value in candidate.support[support_order])
    coefficients = tuple(np.round(candidate.coefficients[support_order], decimals=8).ravel())
    return (*support, *coefficients)


def _support_key(candidate: AttackCandidate) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in candidate.support))


def _aligned_coefficients(
    candidate: AttackCandidate,
    target_support: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Align donor coefficients by atom identity instead of arbitrary row position."""
    lookup = {
        int(atom): candidate.coefficients[row]
        for row, atom in enumerate(candidate.support)
    }
    aligned = np.asarray(fallback, dtype=np.float64).copy()
    for row, atom in enumerate(target_support):
        if int(atom) in lookup:
            aligned[row] = lookup[int(atom)]
    return aligned


def _update_support_importance(
    base_importance: np.ndarray,
    candidates: Iterable[AttackCandidate],
    attackable_mask: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    values = list(candidates)
    if not values or learning_rate <= 0.0:
        return np.asarray(base_importance, dtype=np.float64).copy()
    ranked = sorted(
        values,
        key=lambda candidate: float(
            candidate.measurements.get(
                "robust_attack_score",
                candidate.measurements.get("robust_abundance_shift", 0.0),
            )
        ),
        reverse=True,
    )
    elite_count = min(len(ranked), max(4, int(np.ceil(len(ranked) * 0.25))))
    elites = ranked[:elite_count]
    scores = np.asarray(
        [float(candidate.measurements["robust_attack_score"]) for candidate in elites],
        dtype=np.float64,
    )
    gains = scores - float(np.min(scores)) + 1e-9
    learned = np.zeros_like(np.asarray(base_importance, dtype=np.float64))
    for candidate, gain in zip(elites, gains):
        learned[candidate.support] += float(gain) / max(len(candidate.support), 1)
    learned[~np.asarray(attackable_mask, dtype=bool)] = 0.0
    if float(learned.max()) > EPS:
        learned /= float(learned.max())
    else:
        learned = np.asarray(base_importance, dtype=np.float64).copy()
    rate = float(np.clip(learning_rate, 0.0, 1.0))
    updated = (1.0 - rate) * np.asarray(base_importance, dtype=np.float64) + rate * learned
    updated[~np.asarray(attackable_mask, dtype=bool)] = 0.0
    updated /= max(float(updated.max()), EPS)
    return updated


def _dominates(first: AttackCandidate, second: AttackCandidate) -> bool:
    assert first.objectives is not None and second.objectives is not None
    return bool(np.all(first.objectives <= second.objectives) and np.any(first.objectives < second.objectives))


def nondominated(candidates: Iterable[AttackCandidate]) -> list[AttackCandidate]:
    unique: dict[tuple[object, ...], AttackCandidate] = {}
    for candidate in candidates:
        unique[_candidate_key(candidate)] = candidate.clone()
    values = list(unique.values())
    archive: list[AttackCandidate] = []
    for index, candidate in enumerate(values):
        if not any(
            other_index != index and _dominates(other, candidate)
            for other_index, other in enumerate(values)
        ):
            archive.append(candidate.clone())
    archive.sort(key=lambda item: float(item.objectives[1]))
    return archive


def _sample_support(
    rng: np.random.Generator,
    atoms: int,
    count: int,
    importance: np.ndarray,
    guided: bool,
    exploration_probability: float,
    attackable_mask: np.ndarray,
) -> np.ndarray:
    valid = np.flatnonzero(np.asarray(attackable_mask, dtype=bool))
    if count > len(valid):
        raise ValueError("attack_atoms cannot exceed the attackable library size")
    if guided:
        probabilities = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
        probabilities[~np.asarray(attackable_mask, dtype=bool)] = 0.0
        if probabilities.sum() <= EPS:
            probabilities = np.asarray(attackable_mask, dtype=np.float64)
            probabilities /= probabilities.sum()
        else:
            probabilities /= probabilities.sum()
        exploration = float(np.clip(exploration_probability, 0.0, 1.0))
        uniform = np.asarray(attackable_mask, dtype=np.float64)
        uniform /= uniform.sum()
        probabilities = (1.0 - exploration) * probabilities + exploration * uniform
        return rng.choice(atoms, size=count, replace=False, p=probabilities).astype(np.int64)
    return rng.choice(valid, size=count, replace=False).astype(np.int64)


def _initial_candidate(
    rng: np.random.Generator,
    library_atoms: int,
    importance: np.ndarray,
    config: SpectralAttackConfig,
    guided: bool,
    attackable_mask: np.ndarray,
) -> AttackCandidate:
    support = _sample_support(
        rng,
        library_atoms,
        config.attack_atoms,
        importance,
        guided,
        config.support_exploration_probability,
        attackable_mask,
    )
    direction = rng.normal(size=(config.attack_atoms, config.basis_dimension))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), EPS)
    amplitude = rng.uniform(0.05, 1.0, size=(config.attack_atoms, 1))
    coefficients = direction * amplitude * config.linf_cap
    return AttackCandidate(support=support, coefficients=coefficients)


def _offspring(
    rng: np.random.Generator,
    current: AttackCandidate,
    first: AttackCandidate,
    second: AttackCandidate,
    importance: np.ndarray,
    config: SpectralAttackConfig,
    library_atoms: int,
    attackable_mask: np.ndarray,
) -> AttackCandidate:
    support = current.support.copy()
    coefficients = current.coefficients.copy()
    if rng.random() < config.atom_swap_probability:
        swap_count = 1
        if config.attack_atoms > 2 and rng.random() < config.multi_atom_swap_probability:
            swap_count = min(config.attack_atoms, max(2, int(np.ceil(config.attack_atoms / 3))))
        rows = np.atleast_1d(
            rng.choice(config.attack_atoms, size=swap_count, replace=False)
        )
        for row_value in rows:
            row = int(row_value)
            excluded = set(int(value) for value in support)
            probabilities = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
            probabilities[~np.asarray(attackable_mask, dtype=bool)] = 0.0
            for value in excluded:
                probabilities[value] = 0.0
            if probabilities.sum() <= 0.0:
                choices = [
                    value
                    for value in np.flatnonzero(attackable_mask)
                    if int(value) not in excluded
                ]
                replacement = int(rng.choice(choices))
            else:
                probabilities /= probabilities.sum()
                exploration = float(
                    np.clip(config.support_exploration_probability, 0.0, 1.0)
                )
                available = np.asarray(attackable_mask, dtype=np.float64).copy()
                for value in excluded:
                    available[value] = 0.0
                available /= available.sum()
                probabilities = (
                    (1.0 - exploration) * probabilities + exploration * available
                )
                replacement = int(rng.choice(library_atoms, p=probabilities))
            support[row] = replacement
            coefficients[row] = rng.normal(size=config.basis_dimension)
            coefficients[row] /= max(float(np.linalg.norm(coefficients[row])), EPS)
            coefficients[row] *= rng.uniform(0.05, 1.0) * config.linf_cap

    first_aligned = _aligned_coefficients(first, support, coefficients)
    second_aligned = _aligned_coefficients(second, support, coefficients)
    proposal = coefficients + config.differential_weight * (
        first_aligned - second_aligned
    )
    crossover = rng.random(coefficients.shape) < config.crossover_rate
    if not np.any(crossover):
        crossover.flat[int(rng.integers(crossover.size))] = True
    coefficients = np.where(crossover, proposal, coefficients)
    mutate = rng.random(coefficients.shape) < config.coefficient_mutation_probability
    coefficients += mutate * rng.normal(
        scale=config.coefficient_mutation_scale * config.linf_cap,
        size=coefficients.shape,
    )
    restart_rows = rng.random(config.attack_atoms) < config.coefficient_restart_probability
    for row in np.flatnonzero(restart_rows):
        direction = rng.normal(size=config.basis_dimension)
        direction /= max(float(np.linalg.norm(direction)), EPS)
        coefficients[int(row)] = (
            direction * rng.uniform(0.05, 1.0) * config.linf_cap
        )
    coefficients = np.clip(coefficients, -config.linf_cap, config.linf_cap)
    return AttackCandidate(support=support, coefficients=coefficients)


def _scalar_value(
    objectives: np.ndarray,
    weights: np.ndarray,
    ideal: np.ndarray,
    scale: np.ndarray,
) -> float:
    normalized = np.abs((objectives - ideal) / np.maximum(scale, 1e-9))
    return float(np.max(weights * normalized))


def select_report_candidate(
    archive: list[AttackCandidate], config: SpectralAttackConfig
) -> AttackCandidate:
    if not archive:
        raise RuntimeError("The Pareto archive is empty")
    threshold = (
        config.sam_cap_radians
        if config.report_sam_threshold is None
        else config.report_sam_threshold
    )
    eligible = [
        candidate
        for candidate in archive
        if float(candidate.measurements["mean_sam_radians"]) <= threshold + 1e-12
    ]
    if not eligible:
        eligible = archive
    return max(
        eligible,
        key=lambda item: float(
            item.measurements.get(
                "robust_attack_score", item.measurements["robust_abundance_shift"]
            )
        ),
    ).clone()


def run_moead_attack(
    evaluator: ProxyAttackEvaluator,
    config: SpectralAttackConfig,
    progress: Callable[[dict[str, float | int]], None] | None = None,
) -> SpectralAttackResult:
    if config.population_size < 3:
        raise ValueError("MOEA/D requires a population of at least three")
    rng = np.random.default_rng(config.seed)
    base_importance = evaluator.importance()
    importance = base_importance.copy()
    atoms = evaluator.scene.library.shape[1]
    population: list[AttackCandidate] = []
    search_candidates: list[AttackCandidate] = []
    while len(population) < config.population_size:
        query_count = evaluator.candidate_queries
        candidate = evaluator.evaluate(
            _initial_candidate(
                rng,
                atoms,
                importance,
                config,
                guided=(len(population) % 3 != 0),
                attackable_mask=evaluator.attackable_mask,
            )
        )
        if evaluator.candidate_queries > query_count:
            population.append(candidate)
            search_candidates.append(candidate.clone())
    weight_axis = np.linspace(
        np.clip(config.moead_min_attack_weight, 0.001, 0.999),
        0.999,
        config.population_size,
    )
    weights = np.stack((weight_axis, 1.0 - weight_axis), axis=1)
    weight_distances = np.linalg.norm(weights[:, None] - weights[None, :], axis=2)
    neighbor_count = min(max(2, config.neighbor_size), config.population_size)
    neighborhoods = np.argsort(weight_distances, axis=1)[:, :neighbor_count]
    archive = nondominated(population)
    history: list[dict[str, float | int]] = []
    best_seen = max(
        float(candidate.measurements["robust_attack_score"])
        for candidate in archive
    )
    stagnant_generations = 0

    for generation in range(config.generations):
        is_stagnant = stagnant_generations >= max(1, config.stagnation_generations)
        generation_config = (
            replace(
                config,
                random_immigrant_probability=max(
                    config.random_immigrant_probability,
                    config.stagnation_immigrant_probability,
                ),
                coefficient_restart_probability=max(
                    config.coefficient_restart_probability,
                    config.stagnation_coefficient_restart_probability,
                ),
                multi_atom_swap_probability=max(
                    config.multi_atom_swap_probability,
                    config.stagnation_multi_atom_swap_probability,
                ),
            )
            if is_stagnant
            else config
        )
        generation_replacements = 0
        for index in rng.permutation(config.population_size):
            neighborhood = neighborhoods[index]
            parent_pool = neighborhood if len(neighborhood) >= 2 else np.arange(config.population_size)
            first_index, second_index = rng.choice(parent_pool, size=2, replace=False)
            use_immigrant = (
                rng.random() < generation_config.random_immigrant_probability
            )
            query_count = evaluator.candidate_queries
            for attempt in range(32):
                if use_immigrant or attempt > 0:
                    child = _initial_candidate(
                        rng,
                        atoms,
                        importance,
                        generation_config,
                        guided=(rng.random() < 0.8),
                        attackable_mask=evaluator.attackable_mask,
                    )
                else:
                    child = _offspring(
                        rng,
                        population[index],
                        population[int(first_index)],
                        population[int(second_index)],
                        importance,
                        generation_config,
                        atoms,
                        evaluator.attackable_mask,
                    )
                child = evaluator.evaluate(child)
                if evaluator.candidate_queries > query_count:
                    break
            else:
                raise RuntimeError("Could not generate a unique MOEA/D candidate")
            search_candidates.append(child.clone())
            all_objectives = np.stack(
                [candidate.objectives for candidate in population] + [child.objectives]
            )
            ideal = np.min(all_objectives, axis=0)
            scale = np.max(all_objectives, axis=0) - ideal
            child_support = _support_key(child)
            support_copies = sum(
                _support_key(candidate) == child_support for candidate in population
            )
            replacements = 0
            for neighbor in neighborhood:
                incumbent = population[int(neighbor)]
                if _scalar_value(child.objectives, weights[int(neighbor)], ideal, scale) <= _scalar_value(
                    incumbent.objectives, weights[int(neighbor)], ideal, scale
                ):
                    incumbent_support = _support_key(incumbent)
                    if (
                        incumbent_support != child_support
                        and support_copies >= config.max_support_copies
                    ):
                        continue
                    population[int(neighbor)] = child.clone()
                    if incumbent_support != child_support:
                        support_copies += 1
                    replacements += 1
                    generation_replacements += 1
                    if replacements >= max(1, config.max_neighbor_replacements):
                        break
            archive = nondominated([*archive, child])
        generation_best = max(
            float(item.measurements["robust_attack_score"]) for item in archive
        )
        if generation_best > best_seen + 1e-9:
            best_seen = generation_best
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        importance = _update_support_importance(
            importance,
            [*population, *archive],
            evaluator.attackable_mask,
            config.support_learning_rate,
        )
        record: dict[str, float | int] = {
            "generation": generation + 1,
            "candidate_queries": evaluator.candidate_queries,
            "archive_size": len(archive),
            "best_robust_abundance_shift": max(
                float(item.measurements["robust_abundance_shift"]) for item in archive
            ),
            "best_attack_score": max(
                float(
                    item.measurements.get(
                        "robust_attack_score",
                        item.measurements["robust_abundance_shift"],
                    )
                )
                for item in archive
            ),
            "minimum_mean_sam_radians": min(
                float(item.measurements["mean_sam_radians"]) for item in archive
            ),
            "unique_population_supports": len(
                {_support_key(candidate) for candidate in population}
            ),
            "neighbor_replacements": generation_replacements,
            "stagnant_generations": stagnant_generations,
            "stagnation_mode": int(is_stagnant),
            "random_immigrant_probability": generation_config.random_immigrant_probability,
            "coefficient_restart_probability": generation_config.coefficient_restart_probability,
            "multi_atom_swap_probability": generation_config.multi_atom_swap_probability,
        }
        history.append(record)
        if progress is not None:
            progress(record)

    selected = select_report_candidate(archive, config)
    attacked, delta, _ = construct_attacked_library(
        evaluator.scene.library, selected, evaluator.basis, config
    )
    return SpectralAttackResult(
        selected=selected,
        attacked_library=attacked,
        delta=delta,
        pareto_archive=archive,
        search_candidates=search_candidates,
        candidate_queries=evaluator.candidate_queries,
        solver_queries=evaluator.solver_queries,
        clean_proxy_diagnostics=evaluator.diagnostics(),
        history=history,
    )
