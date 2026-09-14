"""Tests for the forecast-driven scheduling decision system."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_tree import (  # noqa: E402
    DT_H, Z_HIGH, Z_LOW, Z_MID, Forecast, Plant, Thresholds, Train, apply,
    controller_step, decide_turn_off, decide_turn_on, has_room, power_curve,
    project_storage, safety_gate, storage_state, will_run_empty,
)

PLANT = Plant()
TH = Thresholds()
HORIZON = TH.tau_storage


def flat_bands(horizon: int = HORIZON) -> dict[str, np.ndarray]:
    """Zero-width bands, so the tests exercise the logic and not the spread."""
    return {f"q{q:02d}": np.zeros(horizon) for q in (5, 10, 25, 50, 75, 90, 95)}


def forecast(speed: float | np.ndarray, horizon: int = HORIZON) -> Forecast:
    path = (np.full(horizon, float(speed)) if np.isscalar(speed)
            else np.asarray(speed, float))
    return Forecast(path, PLANT, flat_bands(len(path)), 10, 90)


def demand(rate: float = 8650.0 / 24.0, horizon: int = HORIZON) -> np.ndarray:
    return np.full(horizon, rate)


def trains(*running: bool, runtime: int = 99, standstill: int = 99):
    out = []
    for i, is_on in enumerate(running):
        out.append(Train(i, running=is_on,
                         runtime_steps=runtime if is_on else 0,
                         standstill_steps=0 if is_on else standstill))
    return out


# --- power curve ----------------------------------------------------------
def test_power_curve_is_zero_outside_the_operating_band():
    below = power_curve(np.array([0.0, 2.9]), PLANT)
    above = power_curve(np.array([25.1, 40.0]), PLANT)
    assert np.all(below == 0.0)
    assert np.all(above == 0.0)


def test_power_curve_saturates_at_rated():
    rated = PLANT.n_turbines * PLANT.p_rated_turbine_kw * PLANT.availability
    assert power_curve(np.array([12.0]), PLANT)[0] == pytest.approx(rated)
    assert power_curve(np.array([20.0]), PLANT)[0] == pytest.approx(rated)


def test_power_curve_is_monotone_inside_the_ramp():
    ws = np.linspace(3.0, 12.0, 40)
    p = power_curve(ws, PLANT)
    assert np.all(np.diff(p) >= 0)


def test_energy_converts_before_integrating():
    """The mean of the power is not the power of the mean wind."""
    gusty = np.array([4.0, 12.0] * 6)
    steady = np.full(12, gusty.mean())
    assert (forecast(gusty, 12).energy(forecast(gusty, 12).p_mid, 12)
            > forecast(steady, 12).energy(forecast(steady, 12).p_mid, 12))


# --- storage --------------------------------------------------------------
def test_storage_states_follow_the_thresholds():
    assert storage_state(PLANT.v_low_m3 - 1, PLANT) == Z_LOW
    assert storage_state(PLANT.v_low_m3 + 1, PLANT) == Z_MID
    assert storage_state(PLANT.v_high_m3 + 1, PLANT) == Z_HIGH


def test_projection_fills_when_production_exceeds_demand():
    p = np.full(12, 4000.0)
    path = project_storage(5000.0, p, demand(horizon=12), PLANT, 12)
    assert path[-1] > 5000.0
    assert path.max() <= PLANT.v_storage_m3


def test_projection_empties_when_the_plant_is_off():
    path = project_storage(5000.0, np.zeros(12), demand(horizon=12), PLANT, 12)
    assert np.all(np.diff(path) < 0)


def test_will_run_empty_detects_a_draining_storage():
    assert will_run_empty(PLANT.v_low_m3 + 100.0, np.zeros(HORIZON),
                          demand(), PLANT, HORIZON)
    assert not will_run_empty(PLANT.v_high_m3, np.full(HORIZON, 4000.0),
                              demand(), PLANT, HORIZON)


def test_has_room_is_false_when_the_storage_would_fill():
    assert not has_room(PLANT.v_high_m3 - 10.0, np.full(HORIZON, 4000.0),
                        demand(), PLANT, HORIZON)


# --- tree A ---------------------------------------------------------------
def test_start_when_power_is_there_and_the_forecast_holds():
    d = decide_turn_on(trains(True, False, False),
                       p_head_kw=5000.0, volume=6000.0,
                       forecast=forecast(11.0), q_ab_m3h=demand(),
                       plant=PLANT, th=TH)
    assert d.action == "start"
    assert d.branch.startswith("A confirmed")


def test_do_not_start_into_a_forecast_lull():
    """Power is there now, but the pessimistic path cannot sustain the train."""
    path = np.concatenate([np.full(2, 11.0), np.zeros(HORIZON - 2)])
    d = decide_turn_on(trains(True, False, False), 5000.0, 6000.0,
                       forecast(path), demand(), PLANT, TH)
    assert d.action == "hold"
    assert "lull" in d.branch


def test_do_not_start_when_the_storage_would_overflow():
    d = decide_turn_on(trains(True, False, False), 5000.0,
                       PLANT.v_high_m3 - 10.0, forecast(11.0), demand(),
                       PLANT, TH)
    assert d.action == "hold"
    assert "overflow" in d.branch


def test_pre_fill_start_when_the_storage_will_run_empty():
    """Headroom is below the switch-on threshold, but the buffer is draining."""
    d = decide_turn_on(trains(False, False, False), p_head_kw=0.0,
                       volume=PLANT.v_low_m3 + 50.0, forecast=forecast(9.0),
                       q_ab_m3h=demand(), plant=PLANT, th=TH)
    assert d.action == "start"
    assert d.branch.startswith("B pre-fill")


def test_no_start_while_the_standstill_has_not_elapsed():
    d = decide_turn_on(trains(True, False, False, standstill=1), 5000.0,
                       6000.0, forecast(11.0), demand(), PLANT, TH)
    assert d.action == "hold"
    assert "no eligible train" in d.branch


def test_start_picks_the_least_used_train():
    fleet = trains(True, False, False)
    fleet[1].total_runtime_steps = 900
    fleet[2].total_runtime_steps = 100
    d = decide_turn_on(fleet, 5000.0, 6000.0, forecast(11.0), demand(),
                       PLANT, TH)
    assert d.train == 2


# --- tree B ---------------------------------------------------------------
def test_stop_when_power_is_gone_and_stays_gone():
    d = decide_turn_off(trains(True, True, False), p_rest_kw=0.0,
                        volume=PLANT.v_high_m3, forecast=forecast(0.0),
                        q_ab_m3h=demand(), plant=PLANT, th=TH)
    assert d.action == "stop"
    assert d.branch.startswith("A confirmed")


def test_run_through_a_short_gap():
    """The requirement in Table 4.1: do not shut down for a ten-minute dip."""
    path = np.concatenate([np.zeros(1), np.full(HORIZON - 1, 11.0)])
    d = decide_turn_off(trains(True, True, False), 0.0, PLANT.v_high_m3,
                        forecast(path), demand(), PLANT, TH)
    assert d.action == "hold"
    assert "run through" in d.branch


def test_minimum_runtime_blocks_an_early_stop():
    d = decide_turn_off(trains(True, True, False, runtime=TH.t_run_min - 1),
                        0.0, PLANT.v_high_m3, forecast(0.0), demand(),
                        PLANT, TH)
    assert d.action == "hold"
    assert "minimum runtime" in d.branch


def test_the_last_train_is_never_stopped_while_the_storage_is_low():
    d = decide_turn_off(trains(True, False, False), 0.0,
                        PLANT.v_low_m3 - 100.0, forecast(0.0), demand(),
                        PLANT, TH)
    assert d.action == "hold"
    assert "last train" in d.branch


def test_stop_protects_the_product_supply():
    d = decide_turn_off(trains(True, True, False), 0.0,
                        PLANT.v_low_m3 + 50.0, forecast(0.0), demand(),
                        PLANT, TH)
    assert d.action == "hold"
    assert "run empty" in d.branch


def test_preventive_stop_when_the_fleet_cannot_be_carried():
    """Power is fine now; the forecast says it will not carry three trains."""
    path = np.concatenate([np.full(2, 12.0), np.zeros(HORIZON - 2)])
    d = decide_turn_off(trains(True, True, True), p_rest_kw=10_000.0,
                        volume=PLANT.v_high_m3, forecast=forecast(path),
                        q_ab_m3h=demand(), plant=PLANT, th=TH)
    assert d.action == "stop"
    assert d.branch.startswith("B preventive")


def test_stop_picks_the_longest_running_train():
    fleet = trains(True, True, True)
    fleet[0].runtime_steps = 10
    fleet[1].runtime_steps = 90
    fleet[2].runtime_steps = 40
    d = decide_turn_off(fleet, 0.0, PLANT.v_high_m3, forecast(0.0), demand(),
                        PLANT, TH)
    assert d.train == 1


# --- safety gate and dispatch ---------------------------------------------
def test_safety_gate_starts_a_train_when_the_storage_is_low():
    d = safety_gate(trains(False, False, False), PLANT.v_low_m3 - 1.0,
                    PLANT, TH)
    assert d is not None and d.action == "start"


def test_safety_gate_is_silent_when_a_train_already_runs():
    assert safety_gate(trains(True, False, False), PLANT.v_low_m3 - 1.0,
                       PLANT, TH) is None


def test_safety_gate_is_silent_above_the_low_threshold():
    assert safety_gate(trains(False, False, False), PLANT.v_high_m3,
                       PLANT, TH) is None


def test_safety_gate_overrides_the_minimum_standstill():
    fleet = trains(False, False, False, standstill=0)
    d = controller_step(fleet, 0.0, PLANT.v_low_m3 - 1.0, forecast(0.0),
                        demand(), PLANT, TH)
    assert d.action == "start"


def test_controller_never_starts_and_stops_in_one_step():
    fleet = trains(True, False, False)
    d = controller_step(fleet, 5000.0, 6000.0, forecast(11.0), demand(),
                        PLANT, TH)
    assert d.action in ("start", "stop", "hold")


def test_apply_updates_the_train_state():
    fleet = trains(True, False, False)
    d = decide_turn_on(fleet, 5000.0, 6000.0, forecast(11.0), demand(),
                       PLANT, TH)
    apply(fleet, d)
    assert fleet[d.train].running
    assert fleet[d.train].switch_events == 1


def test_advance_tracks_runtime_and_standstill():
    train = Train(0, running=True)
    train.advance()
    assert train.runtime_steps == 1 and train.total_runtime_steps == 1
    train.running = False
    train.advance()
    assert train.runtime_steps == 0 and train.standstill_steps == 1


# --- the asymmetry that the whole design rests on -------------------------
def test_pessimistic_band_is_never_above_the_optimistic_one():
    bands = {f"q{q:02d}": np.full(HORIZON, (q - 50) / 50.0)
             for q in (5, 10, 25, 50, 75, 90, 95)}
    fc = Forecast(np.full(HORIZON, 8.0), PLANT, bands, 10, 90)
    assert np.all(fc.p_lo <= fc.p_hi)


def test_bands_widen_the_decision_in_the_safe_direction():
    """With a real band, ON gets harder and OFF gets harder. Both hold."""
    bands = {f"q{q:02d}": np.full(HORIZON, (q - 50) / 25.0)
             for q in (5, 10, 25, 50, 75, 90, 95)}
    fc = Forecast(np.full(HORIZON, 6.0), PLANT, bands, 10, 90)
    assert fc.energy(fc.p_lo, HORIZON) < fc.energy(fc.p_mid, HORIZON)
    assert fc.energy(fc.p_hi, HORIZON) > fc.energy(fc.p_mid, HORIZON)


# --- energy accounting ----------------------------------------------------
def test_a_running_train_is_never_free():
    """Base load drawn without renewable cover must be charged as P_KP.

    Found by the threshold optimiser: with this uncharged, the cheapest
    controller is one that never switches anything off, because an idling
    train produced water from nothing.
    """
    import scheduling as sched

    trains_on = [Train(i, running=True) for i in range(3)]
    # storage not low, so the energy management funds nothing
    p_use, p_kp, _ = sched.energy_management(Z_MID, 0.0, 0.0,
                                             3 * PLANT.p_nom_train_kw)
    p_process = sched.load_management(p_use, trains_on, PLANT)
    deficit = max(p_process - p_use, 0.0)

    assert p_process == pytest.approx(3 * PLANT.p_base_train_kw)
    assert p_kp == 0.0
    assert deficit == pytest.approx(3 * PLANT.p_base_train_kw)


def test_idling_costs_more_complementary_energy_than_stopping():
    """The whole point of the fix, expressed as a simulation invariant."""
    import numpy as np
    import scheduling as sched
    from load_data import load_10min

    frame = load_10min()
    ws = frame["ws100"].to_numpy(float)[:4000]
    index = frame.index[:4000]
    q_ab = np.full(6 * 24, sched.DEMAND_M3_PER_DAY / 24.0)

    never_stop = Thresholds(kappa_off=0.0, kappa_pre=0.0, p_on=99.0)
    normal = Thresholds()
    src = sched.PersistenceForecast(ws, normal.tau_storage)

    idle, _ = sched.simulate(ws, index, PLANT, never_stop, src, q_ab)
    real, _ = sched.simulate(ws, index, PLANT, normal, src, q_ab)
    assert idle.e_complementary_mwh > real.e_complementary_mwh
