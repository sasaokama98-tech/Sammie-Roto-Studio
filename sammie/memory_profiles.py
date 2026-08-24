"""Matting memory/performance profiles for VFX workloads."""

from __future__ import annotations

from dataclasses import dataclass


MEMORY_SAFE = "Memory Safe"
BALANCED = "Balanced"
FAST = "Fast"
CUSTOM = "Custom"
PROFILE_NAMES = (MEMORY_SAFE, BALANCED, FAST, CUSTOM)


@dataclass(frozen=True)
class MemoryProfile:
    name: str
    description: str
    settings: dict[str, int | str]


_PROFILES = {
    MEMORY_SAFE: MemoryProfile(
        name=MEMORY_SAFE,
        description="Lowest practical VRAM usage with smaller tiles and batches.",
        settings={
            "matany_res": 480,
            "matany_overlap": 2,
            "matany_chunk": 16,
            "vitmatte_roi_margin": 48,
            "vitmatte_tile_size": 768,
            "vitmatte_tile_overlap": 96,
            "mematte_roi_margin": 64,
            "mematte_tile_size": 1024,
            "mematte_tile_overlap": 96,
            "mematte_max_tokens": 6144,
            "mematte_precision": "Float16",
            "hybrid_flow_resolution": 480,
        },
    ),
    BALANCED: MemoryProfile(
        name=BALANCED,
        description="Recommended balance of edge quality, throughput, and VRAM.",
        settings={
            "matany_res": 720,
            "matany_overlap": 2,
            "matany_chunk": 16,
            "vitmatte_roi_margin": 64,
            "vitmatte_tile_size": 1024,
            "vitmatte_tile_overlap": 128,
            "mematte_roi_margin": 96,
            "mematte_tile_size": 2048,
            "mematte_tile_overlap": 128,
            "mematte_max_tokens": 12000,
            "mematte_precision": "Float16",
            "hybrid_flow_resolution": 720,
        },
    ),
    FAST: MemoryProfile(
        name=FAST,
        description="Higher throughput and detail with substantially more VRAM.",
        settings={
            "matany_res": 1080,
            "matany_overlap": 2,
            "matany_chunk": 32,
            "vitmatte_roi_margin": 96,
            "vitmatte_tile_size": 2048,
            "vitmatte_tile_overlap": 128,
            "mematte_roi_margin": 128,
            "mematte_tile_size": 3072,
            "mematte_tile_overlap": 128,
            "mematte_max_tokens": 18000,
            "mematte_precision": "Float16",
            "hybrid_flow_resolution": 1080,
        },
    ),
}


def get_memory_profile(name: str) -> MemoryProfile | None:
    """Return a named built-in profile, or None for Custom/unknown values."""

    return _PROFILES.get(name)


def apply_memory_profile(settings_manager, name: str) -> dict[str, int | str]:
    """Apply a built-in profile to session settings and return its values."""

    profile = get_memory_profile(name)
    if profile is None:
        settings_manager.set_session_setting("memory_profile", CUSTOM)
        return {}
    settings_manager.set_session_setting("memory_profile", profile.name)
    for key, value in profile.settings.items():
        settings_manager.set_session_setting(key, value)
    return dict(profile.settings)
