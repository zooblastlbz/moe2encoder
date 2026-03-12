"""Trainer package for Step1/Step2/Step3 pipelines.

Modules are intentionally imported lazily by callers to avoid package-level
dependencies (for example, Step3 should not require torch at import time).
"""

__all__ = []
