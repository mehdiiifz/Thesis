"""Forecast-driven turn-on / turn-off decision system for the SWRO trains.

Run from the project root:

    python src/decision_tree.py          # demo on the measured wind record

What this is
------------
A drop-in replacement for the on/off switching rule inside MaxPen. It does NOT
touch the energy management: `Energiemanagement`, `calc_P_PLmax`, `calc_P_req`
and the load management keep their current behaviour and still produce P_use,
P_KPreq and P_ver. This module only decides, at each time step, *which train to
start or stop*, and it decides it with the wind forecast rather than with the
instantaneous power alone.

The two trees follow the structure drafted in section 6 of
260806_Draft_Decision_Trees_and_Scheduling, with every threshold named and every
forecast query resolved against a calibrated uncertainty band.

The one design principle worth remembering
------------------------------------------
Every turn-ON query reads the PESSIMISTIC band and every turn-OFF query reads
the OPTIMISTIC band. Both biases point the same way: do not start a train the
wind may not sustain, and do not stop a train the wind may yet carry. That is
the mechanism by which the forecast reduces the on/off count, which is the
requirement in Table 4.1 of the list of requirements.

Storage states
--------------
The MaxPen pseudocode encodes low/middle/high as 0/1/2; the parameter list
handed over with this task used 1/2/3. Only the symbols below are used in the
logic, so either numbering can be plugged in at the boundary without touching
the trees.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_utils import RESULTS, STEP_MINUTES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUANTILE_CSV = RESULTS / "forecast_residual_quantiles.csv"

Z_LOW, Z_MID, Z_HIGH = 0, 1, 2
Z_NAME = {Z_LOW: "low", Z_MID: "middle", Z_HIGH: "high"}

DT_H = STEP_MINUTES / 60.0          # one simulation step, in hours


# ---------------------------------------------------------------------------
# Plant and controller parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Plant:
    """Plant data.

    The wind farm follows the example configuration of AP1 2.2.1 (four 2,3 MW
    turbines). The SWRO side does NOT: that example is sized for a site with a
    far better wind resource than this record, whose capacity factor is 0,25
    and whose mean farm output is 2,3 MW. Kept at 3 x 2,1 MW the plant would
    sit in state low for 57 % of the year, run all three trains 97 % of the
    time and never switch, which tells nothing about a switching rule. The
    train size and the storage below are therefore scaled to this resource,
    keeping the ratios of the original example: nominal plant capacity about
    2,5 times the average demand, storage about 1,5 days.
    """

    n_trains: int = 3
    p_nom_train_kw: float = 1400.0        # nominal electrical power of a train
    baseload_frac: float = 0.40           # base load as a share of nominal
    sec_kwh_per_m3: float = 4.5           # specific energy consumption
    n_turbines: int = 4
    p_rated_turbine_kw: float = 2300.0
    cut_in_ms: float = 3.0                # placeholder IEC-shaped curve, to be
    rated_ms: float = 12.0                # replaced by the manufacturer curve
    cut_out_ms: float = 25.0
    availability: float = 0.97            # wake and availability derating
    v_storage_m3: float = 13000.0         # usable product storage volume
    v_low_m3: float = 2600.0              # low  <-> middle threshold  (20 %)
    v_high_m3: float = 10400.0            # middle <-> high threshold  (80 %)

    @property
    def p_base_train_kw(self) -> float:
        return self.p_nom_train_kw * self.baseload_frac


@dataclass(frozen=True)
class Thresholds:
    """Every tunable of the decision system, in one place.

    These are the parameters a genetic algorithm or a parameter study would
    search over. Defaults reproduce the current MaxPen rule where one exists
    (p_on, p_off) and are first estimates everywhere else.
    """

    p_on: float = 1.30            # start when headroom >= p_on  * base load
    p_off: float = 1.00           # stop  when headroom <  p_off * base load
    t_run_min: int = 6            # minimum runtime, steps (60 min)
    t_off_min: int = 3            # minimum standstill, steps (30 min)
    tau_sustain: int = 6          # a start must be sustainable this long
    tau_off: int = 12             # reactive-stop energy window (120 min)
    tau_pre: int = 24             # preventive-stop energy window (240 min)
    tau_room: int = 12            # storage-overflow lookahead
    tau_storage: int = 48         # storage-emptying lookahead (480 min)
    cover_frac: float = 0.75      # share of tau_sustain that must be covered
    kappa_off: float = 0.50       # penetration target inside tau_off
    kappa_pre: float = 1.00       # base-load cover required inside tau_pre
    q_pessimistic: int = 10       # residual quantile used by every ON query
    q_optimistic: int = 90        # residual quantile used by every OFF query
    max_switches_per_step: int = 1


@dataclass
class Train:
    index: int
    running: bool = False
    runtime_steps: int = 0        # steps since the current start
    standstill_steps: int = 999   # steps since the last stop
    total_runtime_steps: int = 0
    switch_events: int = 0

    def advance(self) -> None:
        if self.running:
            self.runtime_steps += 1
            self.total_runtime_steps += 1
            self.standstill_steps = 0
        else:
            self.standstill_steps += 1
            self.runtime_steps = 0


@dataclass
class Decision:
    """What the controller did, and which branch of the tree produced it."""

    action: str                   # "start", "stop" or "hold"
    train: int | None = None
    branch: str = "-"
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        target = "" if self.train is None else f" train {self.train}"
        return f"{self.action}{target} [{self.branch}]"


# ---------------------------------------------------------------------------
# Forecast layer: wind speed -> power -> decision signals
# ---------------------------------------------------------------------------
def power_curve(ws: np.ndarray, plant: Plant) -> np.ndarray:
    """Wind farm power in kW for a wind speed array in m/s.

    A generic IEC-shaped curve: zero below cut-in, cubic to rated, flat to
    cut-out, zero above. Replace with the manufacturer table when it is
    available; the shape matters more than the exact numbers for the logic
    below, but not for the reported energies.
    """
    ws = np.asarray(ws, dtype=float)
    p = np.zeros_like(ws)
    ramp = (ws >= plant.cut_in_ms) & (ws < plant.rated_ms)
    flat = (ws >= plant.rated_ms) & (ws <= plant.cut_out_ms)
    span = plant.rated_ms ** 3 - plant.cut_in_ms ** 3
    p[ramp] = ((ws[ramp] ** 3 - plant.cut_in_ms ** 3) / span)
    p[flat] = 1.0
    return p * plant.p_rated_turbine_kw * plant.n_turbines * plant.availability


def load_bands(horizon_steps: int) -> dict[str, np.ndarray]:
    """Residual quantiles per lead step, from src/forecast_calibration.py."""
    table = pd.read_csv(QUANTILE_CSV)
    block = table[table["horizon_steps"] == horizon_steps].sort_values("step")
    if block.empty:
        raise ValueError(f"no calibration for horizon {horizon_steps}")
    return {c: block[c].to_numpy(float)
            for c in block.columns if c.startswith("q")}


@dataclass
class Forecast:
    """One issued wind forecast, converted into the signals the trees need.

    `speed` is the point trajectory from the LSTM, one value per ten-minute
    lead step. The pessimistic and optimistic trajectories are formed by adding
    the calibrated residual quantiles, so both inherit the measured asymmetry
    rather than assuming a symmetric Gaussian band.
    """

    speed: np.ndarray
    plant: Plant
    bands: dict[str, np.ndarray]
    q_low: int = 10
    q_high: int = 90

    def _shift(self, key: str) -> np.ndarray:
        return np.maximum(self.speed + self.bands[key][:len(self.speed)], 0.0)

    @property
    def p_mid(self) -> np.ndarray:
        return power_curve(self.speed, self.plant)

    @property
    def p_lo(self) -> np.ndarray:
        """Pessimistic power path. Read by every turn-ON query."""
        return power_curve(self._shift(f"q{self.q_low:02d}"), self.plant)

    @property
    def p_hi(self) -> np.ndarray:
        """Optimistic power path. Read by every turn-OFF query."""
        return power_curve(self._shift(f"q{self.q_high:02d}"), self.plant)

    def energy(self, path: np.ndarray, steps: int) -> float:
        """Forecast energy in kWh over the first `steps` lead steps.

        The conversion to power happens before the integration, never after:
        the power curve is strongly non-linear, so the power of the mean wind
        is not the mean of the power.
        """
        steps = min(steps, len(path))
        return float(path[:steps].sum() * DT_H)


# ---------------------------------------------------------------------------
# Storage forecast
# ---------------------------------------------------------------------------
def process_power(path: np.ndarray, n_on: int, plant: Plant) -> np.ndarray:
    """Power the trains would actually draw if `n_on` of them are running.

    Bounded below by the fleet base load, because when the product storage is
    low the energy management tops the trains up with complementary power, and
    above by the fleet nominal power. Available renewable power beyond that is
    excess, not consumption. Projecting the storage with the raw renewable
    path instead of this would credit the plant with water it never produced.
    """
    if n_on <= 0:
        return np.zeros_like(path)
    return np.clip(path, n_on * plant.p_base_train_kw,
                   n_on * plant.p_nom_train_kw)


def storage_state(volume: float, plant: Plant) -> int:
    if volume <= plant.v_low_m3:
        return Z_LOW
    if volume >= plant.v_high_m3:
        return Z_HIGH
    return Z_MID


def project_storage(volume: float, p_process_kw: np.ndarray,
                    q_ab_m3h: np.ndarray, plant: Plant,
                    steps: int) -> np.ndarray:
    """Product-storage volume over the next `steps` steps.

    q_zu follows from the process power and the specific energy consumption,
    q_ab is the demand profile, which is known rather than forecast.
    """
    steps = min(steps, len(p_process_kw), len(q_ab_m3h))
    q_zu = p_process_kw[:steps] / plant.sec_kwh_per_m3      # m3/h
    delta = (q_zu - q_ab_m3h[:steps]) * DT_H                # m3 per step
    path = volume + np.cumsum(delta)
    return np.clip(path, 0.0, plant.v_storage_m3)


def will_run_empty(volume: float, p_process_kw: np.ndarray,
                   q_ab_m3h: np.ndarray, plant: Plant, steps: int) -> bool:
    path = project_storage(volume, p_process_kw, q_ab_m3h, plant, steps)
    return bool(path.size and path.min() <= plant.v_low_m3)


def has_room(volume: float, p_process_kw: np.ndarray, q_ab_m3h: np.ndarray,
             plant: Plant, steps: int) -> bool:
    """True if the storage does not reach the high threshold within `steps`."""
    path = project_storage(volume, p_process_kw, q_ab_m3h, plant, steps)
    return bool(path.size == 0 or path.max() < plant.v_high_m3)


# ---------------------------------------------------------------------------
# Safety gate, evaluated before either tree
# ---------------------------------------------------------------------------
def safety_gate(trains: list[Train], volume: float, plant: Plant,
                th: Thresholds) -> Decision | None:
    """Enforce the premise the whole MaxPen strategy rests on.

    Complementary energy is used only when the product storage is empty, but
    it can only be used at all if a train is running to consume it. So when the
    storage is low, at least one train must run, whatever the forecast says.
    Without this gate the trees will occasionally let the storage run dry
    during a long lull, because their storage lookahead reaches only as far as
    the forecast horizon.

    Returns None when no override is needed, so the trees run normally.
    """
    if storage_state(volume, plant) != Z_LOW:
        return None
    if any(t.running for t in trains):
        return None
    idle = [t for t in trains if not t.running]
    train = min(idle, key=lambda t: (t.total_runtime_steps, t.index))
    return Decision("start", train.index, "S storage low, demand must be met")


# ---------------------------------------------------------------------------
# Tree A: turn a train ON
# ---------------------------------------------------------------------------
def decide_turn_on(trains: list[Train], p_head_kw: float, volume: float,
                   forecast: Forecast, q_ab_m3h: np.ndarray, plant: Plant,
                   th: Thresholds) -> Decision:
    """Should a further train be started, and which one?

    Gate      candidate must be off and past its minimum standstill
    Node A1   is the current headroom above the switch-on threshold?
    Node A2   does the PESSIMISTIC forecast sustain it for the minimum runtime?
    Node A3   is there room in the storage over tau_room?
    Node A4   (second branch) will the storage run empty within tau_storage?
    Node A5   is renewable power there now and coming?
    """
    candidates = [t for t in trains
                  if not t.running and t.standstill_steps >= th.t_off_min]
    if not candidates:
        return Decision("hold", branch="A0 no eligible train")

    # AP1: start the train with the lowest overall operating time.
    train = min(candidates, key=lambda t: (t.total_runtime_steps, t.index))
    p_base = plant.p_base_train_kw
    n_on = sum(t.running for t in trains)
    p_hold = process_power(forecast.p_lo, n_on, plant)
    p_with_train = process_power(forecast.p_lo, n_on + 1, plant)

    if p_head_kw >= th.p_on * p_base:
        # --- A2 sustain check, pessimistic band ---------------------------
        # Two conditions, because energy alone can be met by a short burst
        # early in the window while the wind collapses right after it.
        window = forecast.p_lo[:th.tau_sustain]
        need = th.p_off * p_base * th.tau_sustain * DT_H
        have = forecast.energy(forecast.p_lo, th.tau_sustain)
        covered = float((window >= th.p_off * p_base).mean()) if window.size else 0.0
        if have < need or covered < th.cover_frac:
            return Decision("hold", branch="A2 lull ahead",
                            detail={"E_lo_kWh": have, "E_need_kWh": need,
                                    "covered": covered})
        # --- A3 room in the storage ---------------------------------------
        if not has_room(volume, p_with_train, q_ab_m3h, plant, th.tau_room):
            return Decision("hold", branch="A3 storage would overflow")
        return Decision("start", train.index, "A confirmed start",
                        detail={"E_lo_kWh": have, "E_need_kWh": need})

    # --- A4 second branch: pre-fill before the storage runs empty ----------
    # Projected with the power the plant would draw if nothing changed, not
    # with the renewable power on offer.
    if will_run_empty(volume, p_hold, q_ab_m3h, plant, th.tau_storage):
        p_now_ok = forecast.p_mid[0] >= p_base
        e_future = forecast.energy(forecast.p_hi, th.tau_storage)
        e_needed = p_base * th.tau_storage * DT_H
        if p_now_ok and e_future >= e_needed:
            if has_room(volume, p_with_train, q_ab_m3h, plant, th.tau_room):
                return Decision("start", train.index, "B pre-fill start",
                                detail={"E_hi_kWh": e_future,
                                        "E_need_kWh": e_needed})
            return Decision("hold", branch="B3 storage would overflow")
        return Decision("hold", branch="B2 no renewable power to use")

    return Decision("hold", branch="A1 headroom below switch-on threshold")


# ---------------------------------------------------------------------------
# Tree B: turn a train OFF
# ---------------------------------------------------------------------------
def decide_turn_off(trains: list[Train], p_rest_kw: float, volume: float,
                    forecast: Forecast, q_ab_m3h: np.ndarray, plant: Plant,
                    th: Thresholds) -> Decision:
    """Should a train be stopped, and which one?

    Gate      candidate must be running and past its minimum runtime
    Node B1   has the remaining power fallen below the switch-off threshold?
    Node B2   does the OPTIMISTIC forecast fail to cover kappa_off of the base
              load over tau_off? If it does not fail, run through the gap.
    Node B3   would the storage run empty if the train stopped?
    Node B4   (second branch) preventive stop: even optimistically, the
              forecast cannot cover the running fleet over tau_pre.
    """
    running = [t for t in trains if t.running]
    if not running:
        return Decision("hold", branch="B0 no train running")

    if len(running) == 1 and storage_state(volume, plant) == Z_LOW:
        return Decision("hold", branch="B0 last train held, storage low")

    eligible = [t for t in running if t.runtime_steps >= th.t_run_min]
    if not eligible:
        return Decision("hold", branch="B0 minimum runtime not reached",
                        detail={"runtimes": [t.runtime_steps for t in running]})

    # AP1: stop the train with the longest current operating time.
    train = max(eligible, key=lambda t: (t.runtime_steps, -t.index))
    p_base = plant.p_base_train_kw
    p_after = process_power(forecast.p_lo, len(running) - 1, plant)

    if p_rest_kw < th.p_off * p_base:
        # --- B2 integral query, optimistic band ---------------------------
        have = forecast.energy(forecast.p_hi, th.tau_off)
        need = th.kappa_off * p_base * th.tau_off * DT_H
        if have >= need:
            return Decision("hold", branch="B2 run through the gap",
                            detail={"E_hi_kWh": have, "E_need_kWh": need})
        # --- B3 protect the product supply --------------------------------
        if will_run_empty(volume, p_after, q_ab_m3h, plant, th.tau_storage):
            return Decision("hold", branch="B3 storage would run empty")
        return Decision("stop", train.index, "A confirmed stop",
                        detail={"E_hi_kWh": have, "E_need_kWh": need})

    # --- B4 preventive branch ---------------------------------------------
    n_running = len(running)
    have = forecast.energy(forecast.p_hi, th.tau_pre)
    need = th.kappa_pre * p_base * n_running * th.tau_pre * DT_H
    if have < need:
        if will_run_empty(volume, p_after, q_ab_m3h, plant, th.tau_storage):
            return Decision("hold", branch="B5 storage would run empty")
        return Decision("stop", train.index, "B preventive stop",
                        detail={"E_hi_kWh": have, "E_need_kWh": need})

    return Decision("hold", branch="B1 remaining power above threshold")


# ---------------------------------------------------------------------------
# One controller step
# ---------------------------------------------------------------------------
def controller_step(trains: list[Train], p_use_kw: float, volume: float,
                    forecast: Forecast, q_ab_m3h: np.ndarray, plant: Plant,
                    th: Thresholds) -> Decision:
    """Evaluate both trees and apply at most one switch.

    Turn-off is evaluated first: relieving an overloaded plant is more urgent
    than adding capacity, and with p_on > p_off the two branches cannot both
    fire under sane thresholds.
    """
    forced = safety_gate(trains, volume, plant, th)
    if forced is not None:
        return forced

    p_ver = sum(plant.p_base_train_kw for t in trains if t.running)
    p_rest = p_use_kw - p_ver

    off = decide_turn_off(trains, p_rest, volume, forecast, q_ab_m3h, plant, th)
    if off.action == "stop":
        return off

    on = decide_turn_on(trains, p_rest, volume, forecast, q_ab_m3h, plant, th)
    if on.action == "start":
        return on

    # Nothing switched. Report both blocking branches rather than one of them,
    # so a run log shows which query actually held the plant still.
    return Decision("hold", branch=f"{off.branch} | {on.branch}")


def apply(trains: list[Train], decision: Decision) -> None:
    if decision.train is None:
        return
    train = trains[decision.train]
    if decision.action == "start" and not train.running:
        train.running = True
        train.runtime_steps = 0
        train.switch_events += 1
    elif decision.action == "stop" and train.running:
        train.running = False
        train.standstill_steps = 0
        train.switch_events += 1


if __name__ == "__main__":
    print(__doc__)
    print("The decision layer is exercised by src/scheduling.py.")
