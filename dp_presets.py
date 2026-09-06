"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus
dem OR-Demo-Portfolio, siehe gm_presets.py in gmm-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dp_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_points_slider": SettingSpec("n", int, C.DEFAULT_N_POINTS, C.N_POINTS_MIN, C.N_POINTS_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "spread_slider": SettingSpec("spread", float, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "alpha_slider": SettingSpec("alpha", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX),
    "sigma_slider": SettingSpec("sigma", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "truncation_slider": SettingSpec(
        "trunc", int, C.DEFAULT_TRUNCATION, C.TRUNCATION_MIN, C.TRUNCATION_MAX
    ),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(n_points, k, spread, seed, alpha, sigma, truncation):
    try:
        st.query_params["n"] = str(int(n_points))
        st.query_params["k"] = str(int(k))
        st.query_params["spread"] = str(spread)
        st.query_params["seed"] = str(int(seed))
        st.query_params["alpha"] = str(alpha)
        st.query_params["sigma"] = str(sigma)
        st.query_params["trunc"] = str(int(truncation))
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_points_slider"] = p["n_points"]
    st.session_state["k_slider"] = p["k"]
    st.session_state["spread_slider"] = p["spread"]
    st.session_state["alpha_slider"] = p["alpha"]
    st.session_state["sigma_slider"] = p["sigma"]
    st.session_state["truncation_slider"] = p["truncation"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
