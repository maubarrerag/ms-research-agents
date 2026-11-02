"""In-silico models of MS: a virtual mouse and a virtual patient.

What this is
------------
A small mechanistic model (ordinary differential equations) of the main
processes biologists think drive MS, wired so agents can run "experiments"
on virtual animals or patients before (or instead of) touching real data:

  periphery (lymph node / blood)          central nervous system
  ------------------------------          ----------------------
  A   myelin-antigen priming signal  -->  Tc  autoreactive T cells in CNS
  Te  autoreactive effector T cells   ->  Bc  B cells / meningeal follicles
  Tr  regulatory T cells (suppress)       Mg  activated microglia/macrophages
  Bp  autoreactive B cells                Bbb blood-brain barrier leakiness
  E   EBV-infected B cell load            My  myelin integrity (1 = intact)
                                          Ol  myelinating oligodendrocytes
                                          Ax  axonal integrity (1 = intact)

Causal links encoded (each is textbook immunology / MS pathology):
  * antigen (immunization in mice; EBV molecular mimicry and infections in
    humans) primes effector T cells; Tregs suppress them
  * effector cells cross the BBB (alpha4-integrin, S1P-dependent egress)
  * in the CNS, T cells re-activate on microglia and activate them further
  * T cells + activated microglia + B-cell antibodies strip myelin
  * myelin debris drains to lymph nodes -> epitope spreading
  * oligodendrocytes (from OPCs) remyelinate; inflammation blocks OPC
    differentiation; ageing shrinks the OPC pool
  * demyelinated axons exposed to inflammation degenerate (irreversible)
  * CNS B-cell follicles become partly self-sustaining (drives progression)

Three models:
  eae_mouse        MOG35-55 EAE in C57BL/6 mice (chronic), 0-5 clinical score
  cuprizone_mouse  toxic demyelination then recovery: tests remyelination
  virtual_patient  relapsing MS over years with EBV, infections, EDSS 0-10

LIMITS - read before trusting anything
--------------------------------------
Parameters are hand-set so the model reproduces broad, published
qualitative behaviour (see `face_validity()`), NOT fitted to patient data.
A result here means "the mechanism as encoded produces this effect", which
is a way to sharpen or reject a hypothesis, never evidence about real MS.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

STATES = ["A", "Te", "Tr", "Bp", "E", "Tc", "Trc", "Bc", "Mg", "Bbb", "My", "Ol", "Ax"]
IX = {s: i for i, s in enumerate(STATES)}
STATE_MEANING = {
    "A": "myelin-antigen priming signal in lymph nodes", "Te": "peripheral autoreactive effector T cells",
    "Tr": "regulatory T cells", "Bp": "peripheral autoreactive B cells", "E": "EBV-infected B cell load",
    "Tc": "autoreactive T cells inside the CNS", "Trc": "regulatory T cells inside the CNS (end attacks)", "Bc": "B cells / meningeal follicles in the CNS",
    "Mg": "activated microglia/macrophages (0-1)", "Bbb": "blood-brain barrier leakiness (0-1)",
    "My": "myelin integrity (1 = intact)", "Ol": "myelinating oligodendrocytes (0-1)",
    "Ax": "axonal integrity (1 = intact)",
}

# ---- parameters ------------------------------------------------------------
# name: (value, meaning). Rates are per day; populations are relative units.
PARAMS: dict[str, tuple[float, str]] = {
    "d_A":     (0.30, "clearance of antigen/priming signal"),
    "k_spread": (0.40, "epitope spreading: myelin debris re-primes the periphery"),
    "k_mimic": (0.15, "EBV molecular mimicry (EBNA1~GlialCAM/ANO2/CRYAB) priming strength"),
    "s_T":     (0.002, "thymic output of autoreactive T cells"),
    "r_T":     (0.75, "effector T cell expansion per unit antigen (Th1/Th17 drive)"),
    "K_T":     (1.00, "effector T cell carrying capacity"),
    "k_sup":   (1.20, "Treg suppression of effector T cells"),
    "d_T":     (0.12, "effector T cell death/contraction"),
    "s_R":     (0.03, "Treg production (raised by vitamin D, gut SCFA)"),
    "r_R":     (0.40, "Treg expansion in response to antigen"),
    "K_R":     (0.60, "Treg carrying capacity"),
    "d_R":     (0.10, "Treg turnover"),
    "s_B":     (0.002, "autoreactive B cell production"),
    "r_B":     (0.60, "B cell activation per unit antigen (BTK-dependent)"),
    "d_B":     (0.08, "B cell turnover"),
    "depl_B":  (0.00, "extra B cell killing (anti-CD20)"),
    "r_E":     (0.04, "EBV-infected B cell growth (latency programs)"),
    "K_E":     (1.00, "EBV load ceiling"),
    "k_ctl":   (0.035, "CD8 T cell control of EBV-infected B cells"),
    "depl_E":  (0.00, "extra killing of EBV-infected B cells"),
    "traf":    (0.5, "trafficking into CNS (alpha4-integrin x S1P egress)"),
    "b0":      (0.05, "baseline BBB permeability"),
    "r_loc":   (1.6, "T cell re-activation on CNS antigen-presenting microglia"),
    "d_Tc":    (0.25, "CNS T cell death/exit"),
    "k_sup2":  (2.00, "suppression of CNS T cells by CNS Tregs"),
    "f_Rent":  (0.05, "relative CNS entry of Tregs"),
    "r_Rc":    (2.0, "CNS Treg expansion driven by inflammation (IL-2 from effectors)"),
    "d_Rc":    (0.08, "CNS Treg loss"),
    "f_Bent":  (0.50, "relative CNS entry of B cells vs T cells"),
    "r_Bc":    (0.06, "growth of meningeal B cell follicles (needs T help)"),
    "K_Bc":    (0.50, "follicle size ceiling"),
    "d_Bc":    (0.02, "CNS B cell turnover (slow: compartmentalized)"),
    "depl_Bc": (0.00, "B cell killing inside CNS (poor antibody penetration)"),
    "k_M":     (0.80, "microglia activation by T cells/B cells/debris (BTK-dependent)"),
    "w_B":     (0.50, "weight of B cells/antibody in microglia activation"),
    "w_deb":   (0.25, "weight of myelin debris in microglia activation"),
    "m_age":   (0.00, "age-related baseline microglial activation (smouldering)"),
    "d_M":     (0.4, "return of microglia to homeostasis"),
    "k_bb":    (0.80, "BBB opening by inflammation"),
    "r_bb":    (0.30, "BBB repair"),
    "k_dem":   (1.6, "demyelination by T cells x microglia"),
    "c_ab":    (0.50, "antibody/complement-mediated demyelination by B cells"),
    "k_rem":   (0.10, "remyelination by oligodendrocytes"),
    "k_diff":  (0.06, "OPC -> oligodendrocyte differentiation (clemastine target)"),
    "opc":     (1.00, "OPC pool / fitness (falls with age; metformin restores)"),
    "inh":     (3.00, "inflammation-induced block of OPC differentiation"),
    "k_olk":   (0.60, "oligodendrocyte killing by inflammation"),
    "k_ax":    (0.035, "axon loss when demyelinated and inflamed"),
    "cup":     (0.00, "cuprizone toxicity to oligodendrocytes"),
    "imm":     (0.00, "immunization strength (EAE induction)"),
    "ptx":     (0.00, "pertussis toxin BBB opening (EAE induction)"),
    "inf_rate": (0.00, "infections per year (virtual patient)"),
    "inf_size": (0.25, "priming boost per infection"),
    "E0":      (0.00, "initial EBV-infected B cell load"),
}

# ---- interventions and conditions ------------------------------------------
# Each modifies parameters while active. ("mul", f) multiplies, ("add", v) adds.
# Effect strength scales with dose (1.0 = typical clinical/standard dose).
MODIFIERS: dict[str, dict] = {
    # disease-modifying therapies
    "natalizumab":     {"kind": "drug", "effects": {"traf": ("mul", 0.20)},
                        "mechanism": "anti-alpha4 integrin: blocks lymphocyte entry into CNS"},
    "fingolimod":      {"kind": "drug", "effects": {"traf": ("mul", 0.30), "k_M": ("mul", 0.90)},
                        "mechanism": "S1P receptor modulator: traps lymphocytes in lymph nodes"},
    "anti_cd20":       {"kind": "drug", "effects": {"depl_B": ("add", 0.60), "depl_E": ("add", 0.15),
                                                    "depl_Bc": ("add", 0.01)},
                        "mechanism": "ocrelizumab-like B cell depletion; poor CNS penetration"},
    "btk_inhibitor":   {"kind": "drug", "effects": {"r_B": ("mul", 0.50), "k_M": ("mul", 0.55)},
                        "mechanism": "CNS-penetrant BTK inhibitor: B cell and microglia activation"},
    "interferon_beta": {"kind": "drug", "effects": {"r_T": ("mul", 0.75), "k_bb": ("mul", 0.60)},
                        "mechanism": "dampens T cell activation, stabilises BBB"},
    "anti_il17":       {"kind": "drug", "effects": {"r_T": ("mul", 0.80), "k_M": ("mul", 0.85)},
                        "mechanism": "blocks IL-17 (Th17 effector cytokine)"},
    "ahsct":           {"kind": "event", "mechanism": "immune reset: ablates and rebuilds the immune repertoire",
                        "reset": {"A": 0.05, "Te": 0.02, "Bp": 0.05, "E": 0.10, "Tc": 0.3}, "boost": {"Tr": 2.0},
                        # lasting: rebuilt repertoire has fewer autoreactive memory cells
                        "effects": {"s_T": ("mul", 0.3), "r_T": ("mul", 0.6)}},
    # remyelination / neuroprotection
    "clemastine":      {"kind": "drug", "effects": {"k_diff": ("mul", 2.0)},
                        "mechanism": "muscarinic M1 antagonist: promotes OPC differentiation"},
    "metformin":       {"kind": "drug", "effects": {"opc": ("mul", 1.6)},
                        "mechanism": "AMPK activation rejuvenates aged OPCs"},
    # EBV-directed
    "ebv_antiviral":   {"kind": "drug", "effects": {"r_E": ("mul", 0.50)},
                        "mechanism": "suppresses EBV lytic replication / reactivation"},
    "ebv_t_cells":     {"kind": "drug", "effects": {"k_ctl": ("mul", 2.0)},
                        "mechanism": "EBV-specific T cells (ATA188-like) kill infected B cells"},
    # lifestyle / supplements
    "vitamin_d":       {"kind": "drug", "effects": {"s_R": ("mul", 1.35), "r_T": ("mul", 0.92)},
                        "mechanism": "VDR signalling raises Tregs, tolerogenic dendritic cells"},
    # risk factors (use as conditions on an arm or the whole cohort)
    "hla_drb1_1501":   {"kind": "condition", "effects": {"r_T": ("mul", 1.30), "k_mimic": ("mul", 1.30)},
                        "mechanism": "risk HLA presents myelin/EBV peptides more efficiently"},
    "low_vitamin_d":   {"kind": "condition", "effects": {"s_R": ("mul", 0.75)},
                        "mechanism": "fewer Tregs"},
    "smoking":         {"kind": "condition", "effects": {"r_T": ("mul", 1.10), "k_M": ("mul", 1.15)},
                        "mechanism": "lung irritation activates T cells; oxidative stress"},
    "adolescent_obesity": {"kind": "condition", "effects": {"r_T": ("mul", 1.15), "s_R": ("mul", 0.90)},
                        "mechanism": "leptin-driven pro-inflammatory state, lower vitamin D"},
    "gut_dysbiosis":   {"kind": "condition", "effects": {"r_T": ("mul", 1.20), "s_R": ("mul", 0.80)},
                        "mechanism": "fewer SCFA-producing bacteria: Th17 up, Treg down"},
    "aged":            {"kind": "condition", "effects": {"opc": ("mul", 0.40), "m_age": ("add", 0.03)},
                        "mechanism": "OPC ageing, primed microglia"},
    "ebv_seronegative": {"kind": "condition", "effects": {"E0": ("mul", 0.0)},
                        "mechanism": "never infected with EBV"},
}

MODELS = {
    "eae_mouse": {
        "description": "C57BL/6 mouse immunized with MOG35-55 + CFA, pertussis toxin day 0 and 2. "
                       "Chronic EAE: onset ~day 9-12, peak ~day 15-18, partial recovery.",
        "days": 40, "dt": 0.1, "n_default": 15,
        "params": {"imm": 0.8, "ptx": 0.6, "E0": 0.0, "k_mimic": 0.0, "r_B": 0.25, "f_Bent": 0.1},
        "score": "EAE clinical score 0-5 (1 limp tail, 2 hind limb weakness, 3 paralysis, 5 death)",
        "metrics": ["incidence", "onset_day", "peak_score", "cumulative_score", "final_score",
                    "min_myelin", "final_myelin", "final_axon"],
    },
    "cuprizone_mouse": {
        "description": "0.2% cuprizone diet for 5 weeks (oligodendrocyte death, no T cell attack), "
                       "then normal diet: spontaneous remyelination. Tests remyelination drugs.",
        "days": 49, "dt": 0.1, "n_default": 15,
        "params": {"imm": 0.0, "ptx": 0.0, "E0": 0.0, "k_mimic": 0.0, "s_T": 0.0005, "k_spread": 0.0,
                   "f_Bent": 0.0, "k_diff": 0.15, "k_rem": 0.15, "k_ax": 0.012},
        "cuprizone_until": 35,
        "score": "no clinical score; read myelin",
        "metrics": ["myelin_at_withdrawal", "final_myelin", "remyelination", "final_axon"],
    },
    "virtual_patient": {
        "description": "Adult with relapsing MS, EBV latent infection, ~1.5 infections/year that "
                       "trigger flares; follows the patient for `days` (default 5 years).",
        "days": 1825, "dt": 0.5, "n_default": 60,
        "params": {"imm": 0.0, "ptx": 0.0, "E0": 0.35, "inf_rate": 1.5, "inf_size": 1.0,
                   # human disease is slower and better controlled than induced mouse EAE
                   "r_T": 3.0, "k_sup": 1.6, "k_mimic": 0.05, "k_spread": 0.1, "r_B": 0.25,
                   "f_Bent": 0.2, "k_dem": 0.5, "k_ax": 0.0012},
        "score": "EDSS-like disability 0-10",
        "metrics": ["relapses", "arr", "lesion_load", "final_edss", "edss_change",
                    "final_myelin", "brain_atrophy"],
    },
}

CV = 0.25  # between-subject variability of key parameters (log-normal)
VARIABLE = ["r_T", "k_sup", "s_R", "r_B", "traf", "k_M", "k_dem", "k_diff", "opc", "k_ax", "k_mimic", "r_E", "k_ctl"]


class SimError(ValueError):
    pass


# ---- spec handling ----------------------------------------------------------

def describe() -> dict:
    """Everything an agent needs to design a simulation."""
    return {
        "warning": "Hand-tuned mechanistic model. Results show what the encoded mechanism implies, "
                   "not what happens in real MS. Use it to sharpen/reject hypotheses before real data.",
        "models": {k: {kk: v[kk] for kk in ("description", "days", "score", "metrics")} for k, v in MODELS.items()},
        "states": STATE_MEANING,
        "parameters": {k: {"default": v, "meaning": m} for k, (v, m) in PARAMS.items()},
        "modifiers": {k: {"kind": v["kind"], "mechanism": v["mechanism"],
                          "effects": v.get("effects", {"reset": v.get("reset"), "boost": v.get("boost")})}
                      for k, v in MODIFIERS.items()},
        "spec_example": {
            "model": "eae_mouse", "n_per_arm": 15, "seed": 1,
            "arms": [
                {"name": "vehicle"},
                {"name": "BTKi prophylactic", "interventions": [{"name": "btk_inhibitor", "dose": 1, "start_day": 0}]},
                {"name": "novel: microglia-sparing remyelinator",
                 "interventions": [{"name": "custom", "effects": {"k_diff": 1.5, "inh": 0.5}, "start_day": 15}]},
            ],
            "cohort_conditions": [],
        },
        "spec_rules": [
            "arm 0 is the control every other arm is compared against",
            "interventions: name from modifiers, or 'custom' with effects {param: multiplier}",
            "dose scales the effect (1 = standard); start_day/stop_day set the window (therapeutic vs preventive)",
            "conditions: risk factors applied to an arm (arm.conditions) or the whole cohort (cohort_conditions)",
            "param_overrides: {param: multiplier} on an arm, to encode a new mechanism permanently",
            "same spec + same seed gives identical results; change seed to check robustness",
        ],
    }


def _check_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise SimError("spec must be an object")
    model = spec.get("model")
    if model not in MODELS:
        raise SimError(f"model must be one of {list(MODELS)}")
    m = MODELS[model]
    arms = spec.get("arms") or []
    if not 1 <= len(arms) <= 8:
        raise SimError("give 1-8 arms; arm 0 is the control")
    n = int(spec.get("n_per_arm") or m["n_default"])
    days = int(spec.get("days") or m["days"])
    if not 2 <= n <= 500:
        raise SimError("n_per_arm must be 2-500")
    if not 5 <= days <= 3650 * 2:
        raise SimError("days must be 5-7300")
    for c in spec.get("cohort_conditions") or []:
        _check_mod(c if isinstance(c, str) else c.get("name"))
    for a in arms:
        for c in a.get("conditions") or []:
            _check_mod(c if isinstance(c, str) else c.get("name"))
        for iv in a.get("interventions") or []:
            if iv.get("name") == "custom":
                for p in (iv.get("effects") or {}):
                    if p not in PARAMS:
                        raise SimError(f"custom effect on unknown parameter {p}")
            else:
                _check_mod(iv.get("name"))
        for p in (a.get("param_overrides") or {}):
            if p not in PARAMS:
                raise SimError(f"unknown parameter {p}")
    return {**spec, "n_per_arm": n, "days": days, "seed": int(spec.get("seed", 1))}


def _check_mod(name):
    if name not in MODIFIERS:
        raise SimError(f"unknown modifier {name!r}; choose from {sorted(MODIFIERS)} or 'custom'")


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


# ---- the model --------------------------------------------------------------

@dataclass
class _Window:
    effects: dict          # param -> ("mul"|"add", value) at dose 1
    dose: float
    start: float
    stop: float

    def active(self, t):
        return self.start <= t < self.stop


def _apply(base: dict, windows: list[_Window], t: float) -> dict:
    p = dict(base)
    for w in windows:
        if w.active(t):
            for k, (op, v) in w.effects.items():
                if op == "mul":
                    p[k] = p[k] * (v ** w.dose if v > 0 else max(0.0, 1 - w.dose))
                else:
                    p[k] = p[k] + v * w.dose
    return p


def _deriv(y, p, t):
    A, Te, Tr, Bp, E, Tc, Trc, Bc, Mg, Bbb, My, Ol, Ax = y
    demy = 1 - My
    prime = p["imm"] * (t / 4.0) * np.exp(1 - t / 4.0) + p["k_mimic"] * E + p["k_spread"] * Mg * demy
    dA = prime - p["d_A"] * A
    dTe = p["s_T"] + p["r_T"] * A * Te * (1 - Te / p["K_T"]) - p["k_sup"] * Tr * Te - p["d_T"] * Te
    dTr = p["s_R"] + p["r_R"] * A * Tr * (1 - Tr / p["K_R"]) - p["d_R"] * Tr
    dBp = p["s_B"] + p["r_B"] * (A + 0.5 * E) * Bp * (1 - Bp) - (p["d_B"] + p["depl_B"]) * Bp
    dE = p["r_E"] * E * (1 - E / p["K_E"]) - (p["k_ctl"] + p["depl_E"]) * E
    entry = p["traf"] * (p["b0"] + Bbb)
    dTc = entry * Te + p["r_loc"] * Tc * Mg * (1 - Tc) - p["d_Tc"] * Tc - p["k_sup2"] * Trc * Tc
    dTrc = p["f_Rent"] * entry * Tr + p["r_Rc"] * Trc * Tc * (1 - Trc) - p["d_Rc"] * Trc
    room = np.maximum(0.0, 1 - Bc / p["K_Bc"])
    dBc = ((p["f_Bent"] * entry * Bp + p["r_Bc"] * Bc * Tc / (Tc + 0.2)) * room
           - (p["d_Bc"] + p["depl_Bc"]) * Bc)
    drive = Tc + p["w_B"] * Bc + p["w_deb"] * demy * (1 - Ol) + p["m_age"] + p["cup"]
    dMg = p["k_M"] * drive * (1 - Mg) - p["d_M"] * Mg
    ptx = p["ptx"] * np.exp(-t / 2.0)
    dBbb = (p["k_bb"] * (Tc + 0.5 * Mg) + ptx) * (1 - Bbb) - p["r_bb"] * (Bbb - p["b0"])
    attack = p["k_dem"] * (Tc + p["c_ab"] * Bc) * Mg + 0.8 * p["cup"] * (1 - Ol)
    dMy = -attack * My + p["k_rem"] * Ol * demy
    dOl = (p["k_diff"] * p["opc"] * (1 - Ol) / (1 + p["inh"] * Mg)
           - (p["k_olk"] * Tc * Mg + 4.0 * p["cup"]) * Ol)
    dAx = -p["k_ax"] * demy * (Tc + Mg) * Ax
    return np.array([dA, dTe, dTr, dBp, dE, dTc, dTrc, dBc, dMg, dBbb, dMy, dOl, dAx])


def _score(model, y):
    Tc, My, Ax = y[IX["Tc"]], y[IX["My"]], y[IX["Ax"]]
    if model == "virtual_patient":
        # acute (relapse) part + fixed damage; disability follows axon loss mostly
        return np.clip(1.0 + 3.0 * Tc / (Tc + 0.15) + 3.0 * (1 - My) + 20.0 * (1 - Ax), 0, 10)
    inflam = Tc ** 2 / (Tc ** 2 + 0.15 ** 2)
    return np.clip(5.0 * (0.55 * inflam + 0.5 * (1 - My) + 2.5 * (1 - Ax)), 0, 5)


def _subject_params(model, n, rng, arm_overrides):
    base = {k: v for k, (v, _) in PARAMS.items()}
    base.update(MODELS[model]["params"])
    base.update({k: base[k] * f for k, f in arm_overrides.items()})
    out = {k: np.full(n, float(v)) for k, v in base.items()}
    for k in VARIABLE:
        out[k] = out[k] * rng.lognormal(0.0, CV, n)
    return out


def _mods_to_windows(names_or_ivs, days, default_start=0.0):
    ws, events = [], []
    for iv in names_or_ivs or []:
        iv = {"name": iv} if isinstance(iv, str) else iv
        dose = float(iv.get("dose", 1.0))
        start = float(iv.get("start_day", default_start))
        stop = float(iv.get("stop_day") or days + 1)
        if iv["name"] == "custom":
            eff = {k: ("mul", float(v)) for k, v in (iv.get("effects") or {}).items()}
            ws.append(_Window(eff, dose, start, stop))
            continue
        mod = MODIFIERS[iv["name"]]
        if mod["kind"] == "event":
            events.append((start, mod))
            if mod.get("effects"):
                ws.append(_Window(mod["effects"], dose, start, stop))
        else:
            ws.append(_Window(mod["effects"], dose, start, stop))
    return ws, events


def simulate_arm(model: str, arm: dict, cohort_conditions, n: int, days: int, seed: int) -> dict:
    cfg = MODELS[model]
    dt = cfg["dt"]
    steps = int(round(days / dt))
    # Same seed => same virtual animals in every arm (paired design, like littermates).
    rng = np.random.default_rng(seed)
    p_subj = _subject_params(model, n, rng, arm.get("param_overrides") or {})
    cond_ws, _ = _mods_to_windows(list(cohort_conditions or []) + list(arm.get("conditions") or []), days)
    cond_ws = [_Window(w.effects, w.dose, -1e9, 1e9) for w in cond_ws]   # conditions last all life
    iv_ws, events = _mods_to_windows(arm.get("interventions"), days)
    windows = cond_ws + iv_ws

    p0 = _apply(p_subj, cond_ws, 0.0)
    y = np.zeros((len(STATES), n))
    y[IX["Te"]] = p0["s_T"] / p0["d_T"]
    y[IX["Tr"]] = p0["s_R"] / p0["d_R"]
    y[IX["Bp"]] = p0["s_B"] / p0["d_B"]
    y[IX["E"]] = p0["E0"]
    y[IX["Bbb"]] = p0["b0"]
    y[IX["My"]] = y[IX["Ol"]] = y[IX["Ax"]] = 1.0
    if model == "virtual_patient":   # start from established disease: some primed cells and lesions
        y[IX["A"]] = 0.15
        y[IX["Te"]] = 0.10
        y[IX["Bc"]] = 0.05
        y[IX["My"]] = 0.92
        y[IX["Ax"]] = 0.97

    # infections: Poisson process per subject (same across arms because rng is shared order)
    inf_rng = np.random.default_rng(seed + 7919)
    rate = float(np.mean(p0["inf_rate"])) / 365.0
    infections = inf_rng.random((steps, n)) < rate * dt if rate else None
    inf_size = inf_rng.uniform(0.5, 1.5, (steps, n)) * p0["inf_size"] if rate else None

    cup_until = cfg.get("cuprizone_until")
    record_every = max(1, int(round(1.0 / dt))) if days <= 400 else max(1, int(round(7.0 / dt)))
    times, traj, scores = [], [], []
    tc_series = []
    done_events = set()
    for s in range(steps):
        t = s * dt
        for i, (et, mod) in enumerate(events):
            if i not in done_events and t >= et:
                for st, frac in mod.get("reset", {}).items():
                    y[IX[st]] *= frac
                for st, f in mod.get("boost", {}).items():
                    y[IX[st]] *= f
                done_events.add(i)
        p = _apply(p_subj, windows, t)
        if cup_until is not None:
            p["cup"] = p["cup"] + (0.09 if t < cup_until else 0.0)
        k1 = _deriv(y, p, t)
        k2 = _deriv(y + dt / 2 * k1, p, t + dt / 2)
        k3 = _deriv(y + dt / 2 * k2, p, t + dt / 2)
        k4 = _deriv(y + dt * k3, p, t + dt)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if infections is not None:
            # flares need EBV: reactivation + mimicry amplify the bystander priming of an infection
            e = y[IX["E"]]
            y[IX["A"]] += infections[s] * inf_size[s] * (0.2 + 0.8 * e / (e + 0.15))
            y[IX["E"]] += infections[s] * 0.05 * (e > 0)  # infections reactivate EBV (if present)
        y = np.maximum(y, 0.0)
        for st in ("Mg", "Bbb", "My", "Ol", "Ax", "Bp", "E", "Trc"):
            y[IX[st]] = np.minimum(y[IX[st]], 1.0)
        if s % record_every == 0:
            times.append(t)
            traj.append(y.copy())
            scores.append(_score(model, y))
        tc_series.append(y[IX["Tc"]].copy())
    traj = np.array(traj)              # time x state x subject
    scores = np.array(scores)          # time x subject
    return {"times": np.array(times), "traj": traj, "scores": scores,
            "tc": np.array(tc_series), "dt": dt}


def _metrics(model: str, r: dict, days: int) -> dict[str, np.ndarray]:
    t, traj, sc = r["times"], r["traj"], r["scores"]
    My, Ax = traj[:, IX["My"]], traj[:, IX["Ax"]]
    if model == "eae_mouse":
        sick = sc >= 1.0
        onset = np.where(sick.any(0), t[np.argmax(sick, axis=0)], np.nan)
        return {"incidence": sick.any(0).astype(float), "onset_day": onset,
                "peak_score": sc.max(0), "cumulative_score": sc.sum(0) * (t[1] - t[0]),
                "final_score": sc[-1], "min_myelin": My.min(0), "final_myelin": My[-1], "final_axon": Ax[-1]}
    if model == "cuprizone_mouse":
        i = int(np.searchsorted(t, MODELS[model]["cuprizone_until"]))
        i = min(i, len(t) - 1)
        return {"myelin_at_withdrawal": My[i], "final_myelin": My[-1],
                "remyelination": My[-1] - My[i], "final_axon": Ax[-1]}
    # virtual patient: a relapse = Tc rising above threshold after >= 30 days below it
    tc, dt = r["tc"], r["dt"]
    above = tc > 0.05
    rises = above[1:] & ~above[:-1]
    gap = int(30 / dt)
    relapses = np.zeros(tc.shape[1])
    for j in range(tc.shape[1]):
        idx = np.flatnonzero(rises[:, j])
        last = -gap
        for k in idx:
            if k - last >= gap:
                relapses[j] += 1
            last = k
    years = days / 365.0
    i1 = int(np.searchsorted(t, min(365, days / 2)))
    lesion = (traj[:, IX["Tc"]] * traj[:, IX["Bbb"]]).sum(0) * (t[1] - t[0]) / years
    return {"relapses": relapses, "arr": relapses / years, "lesion_load": lesion,
            "final_edss": sc[-1], "edss_change": sc[-1] - sc[i1],
            "final_myelin": My[-1], "brain_atrophy": 1 - Ax[-1]}


def _compare(ctrl: np.ndarray, x: np.ndarray) -> dict:
    a, b = ctrl[~np.isnan(ctrl)], x[~np.isnan(x)]
    out = {"diff": float(np.mean(b) - np.mean(a)) if len(a) and len(b) else None}
    if len(a) < 2 or len(b) < 2:
        return out
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt((va + vb) / 2) or 1e-12
    out["cohen_d"] = float((b.mean() - a.mean()) / pooled)
    try:
        from scipy import stats
        out["p_welch"] = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
    except Exception:
        se = math.sqrt(va / len(a) + vb / len(b)) or 1e-12
        out["p_welch"] = float(math.erfc(abs((b.mean() - a.mean()) / se) / math.sqrt(2)))
    return out


def run(spec: dict) -> dict:
    """Run a full in-silico experiment. Returns summary stats, arm-vs-control
    comparisons and a per-day mean trajectory per arm."""
    spec = _check_spec(spec)
    model, n, days, seed = spec["model"], spec["n_per_arm"], spec["days"], spec["seed"]
    arm_metrics, curves, summary = [], {}, []
    for a in spec["arms"]:
        r = simulate_arm(model, a, spec.get("cohort_conditions"), n, days, seed)
        m = _metrics(model, r, days)
        arm_metrics.append(m)
        name = a.get("name") or f"arm{len(arm_metrics) - 1}"
        summary.append({"arm": name, "n": n, **{k: {"mean": _r(np.nanmean(v)), "sd": _r(np.nanstd(v, ddof=1))}
                                               for k, v in m.items()}})
        curves[name] = {"day": [_r(x) for x in r["times"]],
                        "score_mean": [_r(x) for x in r["scores"].mean(1)],
                        "myelin_mean": [_r(x) for x in r["traj"][:, IX["My"]].mean(1)],
                        "axon_mean": [_r(x) for x in r["traj"][:, IX["Ax"]].mean(1)],
                        "cns_T_mean": [_r(x) for x in r["traj"][:, IX["Tc"]].mean(1)]}
    comparisons = []
    for i in range(1, len(spec["arms"])):
        comparisons.append({"arm": summary[i]["arm"], "vs": summary[0]["arm"],
                            **{k: {kk: _r(vv) for kk, vv in _compare(arm_metrics[0][k], arm_metrics[i][k]).items()}
                               for k in arm_metrics[0]}})
    return {"model": model, "spec_hash": spec_hash(spec), "spec": spec, "summary": summary,
            "comparisons": comparisons, "curves": curves,
            "note": "In-silico result from a hand-tuned mechanistic model; not evidence about real patients."}


def _r(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, nd)


# ---- face validity -----------------------------------------------------------
# Known, robust published effects the model must reproduce in direction.
BENCHMARKS = [
    ("EAE: BTK inhibitor from day 0 lowers peak score", "eae_mouse",
     [{"name": "vehicle"}, {"name": "btki", "interventions": [{"name": "btk_inhibitor", "start_day": 0}]}],
     "peak_score", "lower"),
    ("EAE: natalizumab-like entry block from day 0 lowers peak score", "eae_mouse",
     [{"name": "vehicle"}, {"name": "nat", "interventions": [{"name": "natalizumab", "start_day": 0}]}],
     "peak_score", "lower"),
    ("EAE: fingolimod from day 0 lowers cumulative score", "eae_mouse",
     [{"name": "vehicle"}, {"name": "fty", "interventions": [{"name": "fingolimod", "start_day": 0}]}],
     "cumulative_score", "lower"),
    ("Cuprizone: clemastine after withdrawal speeds remyelination", "cuprizone_mouse",
     [{"name": "vehicle"}, {"name": "clem", "interventions": [{"name": "clemastine", "start_day": 35}]}],
     "remyelination", "higher"),
    ("Cuprizone: aged mice remyelinate less", "cuprizone_mouse",
     [{"name": "young"}, {"name": "aged", "conditions": ["aged"]}], "remyelination", "lower"),
    ("Cuprizone: metformin rescues remyelination in aged mice", "cuprizone_mouse",
     [{"name": "aged", "conditions": ["aged"]},
      {"name": "aged+metformin", "conditions": ["aged"], "interventions": [{"name": "metformin"}]}],
     "remyelination", "higher"),
    ("Patient: natalizumab cuts relapse rate", "virtual_patient",
     [{"name": "untreated"}, {"name": "nat", "interventions": [{"name": "natalizumab"}]}], "arr", "lower"),
    ("Patient: anti-CD20 cuts relapse rate", "virtual_patient",
     [{"name": "untreated"}, {"name": "cd20", "interventions": [{"name": "anti_cd20"}]}], "arr", "lower"),
    ("Patient: interferon-beta cuts relapses less than natalizumab", "virtual_patient",
     [{"name": "ifn", "interventions": [{"name": "interferon_beta"}]},
      {"name": "nat", "interventions": [{"name": "natalizumab"}]}], "arr", "lower"),
    ("Patient: EBV-seronegative people develop less disease", "virtual_patient",
     [{"name": "ebv+"}, {"name": "ebv-", "conditions": ["ebv_seronegative"]}], "lesion_load", "lower"),
]


def face_validity(n: int = 20, seed: int = 3) -> list[dict]:
    out = []
    for label, model, arms, metric, want in BENCHMARKS:
        res = run({"model": model, "n_per_arm": n, "seed": seed, "arms": arms})
        c = res["comparisons"][0][metric]
        ok = c["diff"] is not None and ((c["diff"] < 0) if want == "lower" else (c["diff"] > 0))
        out.append({"check": label, "metric": metric, "control": res["summary"][0][metric]["mean"],
                    "treated": res["summary"][1][metric]["mean"], "cohen_d": c.get("cohen_d"), "pass": bool(ok)})
    return out
