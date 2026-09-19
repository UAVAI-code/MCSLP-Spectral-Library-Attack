"""Released MCSLP spectral-library poisoning attack."""

from .mcslp_attack import (
    ATTACK_MODEL_VERSION,
    AttackCandidate,
    AttackScene,
    ProxyAttackEvaluator,
    ProxyEstimate,
    SpectralAttackConfig,
    SpectralAttackResult,
    attack_model_version,
    construct_attacked_library,
    low_frequency_cosine_basis,
    run_moead_attack,
    select_report_candidate,
    spectral_angle,
    spectral_roughness,
)

__all__ = [
    "ATTACK_MODEL_VERSION", "AttackCandidate", "AttackScene",
    "ProxyAttackEvaluator", "ProxyEstimate", "SpectralAttackConfig",
    "SpectralAttackResult", "attack_model_version", "construct_attacked_library",
    "low_frequency_cosine_basis", "run_moead_attack", "select_report_candidate",
    "spectral_angle", "spectral_roughness",
]
