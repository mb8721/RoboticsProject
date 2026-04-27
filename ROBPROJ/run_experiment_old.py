"""
run_experiment.py
=================
Experiment runner for the fault-tolerant three-legged locomotion project.
Integrates ThreeLegGaitController with the Pupper hardware / simulation.

Three conditions (per proposal §IV):
  (a) baseline_4leg  – normal 4-leg trot, no fault (control condition)
  (b) three_leg_FR   – 3-leg gait, leg FR disabled at t=3s
  (c) three_leg_HL   – 3-leg gait, leg HL disabled at t=3s
  5 trials per condition, 10 s each.

Usage
-----
# Quick smoke test (no robot, no Pupper sim):
python run_experiment.py --mode smoke

# Full simulation (inside Pupper sim environment):
python run_experiment.py --mode sim --disable-leg FR

# Real Pupper:
python run_experiment.py --mode real --disable-leg FR

# Run all 3 conditions × 5 trials (real robot):
python run_experiment.py --mode real --all-conditions

Integration with Lab 3
----------------------
Search for the two TODO blocks below and fill in your Lab 3 interface.
Everything else is ready to run.
"""

import argparse
import time
import numpy as np

from three_leg_gait import ThreeLegGaitController, FR, FL, HR, HL, LEG_NAMES
from metrics import MetricsLogger

LEG_MAP = {"FR": FR, "FL": FL, "HR": HR, "HL": HL, "NONE": None}
DT = 0.02   # 50 Hz control rate

# ===========================================================================
# TODO BLOCK 1 — Pupper interface wrappers
# Replace these stubs with your Lab 3 robot interface.
# ===========================================================================

def robot_init():
    """
    Initialize and return a handle to the Pupper (real or sim).
    Return None for smoke-test mode.

    Example (fill in your Lab 3 class names):
        from pupper_controller import PupperInterface
        robot = PupperInterface()
        robot.start()
        return robot
    """
    # TODO: replace with your Lab 3 robot initialisation
    print("[Robot] NOTE: using stub robot interface. Fill in robot_init() "
          "with your Lab 3 code before running on hardware.")
    return None


def robot_get_state(robot) -> dict:
    """
    Read sensor data from Pupper.
    Returns dict with keys: vx, vz, roll, pitch, body_height.

    Example:
        state = robot.get_state()
        return {
            "vx":           state.body_vel[0],
            "vz":           state.body_vel[2],
            "roll":         state.orientation_rpy[0],
            "pitch":        state.orientation_rpy[1],
            "body_height":  state.height,
        }
    """
    # TODO: replace with real sensor reads
    # Stub returns plausible values for smoke-test
    return {
        "vx":          0.18 + np.random.normal(0, 0.02),
        "vz":          np.random.normal(0, 0.01),
        "roll":        np.random.normal(0, 0.04),
        "pitch":       np.random.normal(0, 0.04),
        "body_height": 0.135 + np.random.normal(0, 0.003),
    }


def robot_send_foot_positions(robot, foot_pos: np.ndarray):
    """
    Send (4,3) foot positions to Pupper via IK → joint commands.

    Example:
        joint_angles = robot.inverse_kinematics(foot_pos)
        robot.set_joint_angles(joint_angles)
    """
    # TODO: replace with your Lab 3 IK + command send
    pass   # stub does nothing


def robot_stop(robot):
    """Safely stop and park the robot."""
    # TODO: e.g. robot.stand_still(); robot.stop()
    pass

# ===========================================================================
# Core trial runner
# ===========================================================================

def run_trial(mode: str,
              disabled_leg_name: str,
              velocity: float = 0.20,
              trial_duration: float = 10.0,
              disable_at_t: float = 3.0) -> dict:
    """
    Run one trial.

    Parameters
    ----------
    mode              : "smoke" | "sim" | "real"
    disabled_leg_name : "NONE" (4-leg baseline) | "FR" | "FL" | "HR" | "HL"
    velocity          : commanded forward velocity (m/s)
    trial_duration    : how long to run (s)
    disable_at_t      : when to trigger leg failure (s); ignored if NONE
    """
    disabled_leg = LEG_MAP[disabled_leg_name.upper()]
    is_fault_trial = disabled_leg is not None

    label = ("4-leg baseline" if not is_fault_trial
             else f"3-leg [{disabled_leg_name}]")
    print(f"\n{'─'*55}")
    print(f"  TRIAL  |  {label}  |  {mode} mode")
    print(f"{'─'*55}")

    if mode == "real":
        print("  Position robot on flat surface.  Press Enter to start…")
        input()

    # Initialise
    ctrl    = ThreeLegGaitController(dt=DT)
    metrics = MetricsLogger(vx_cmd=velocity, trial_duration=trial_duration)
    robot   = robot_init() if mode in ("sim", "real") else None

    metrics.start_trial()
    t = 0.0
    leg_disabled = False

    try:
        while t < trial_duration:
            loop_t0 = time.time()

            # ---- Trigger leg failure ----
            if is_fault_trial and not leg_disabled and t >= disable_at_t:
                ctrl.disable_leg(disabled_leg)
                leg_disabled = True

            # ---- Controller step ----
            foot_pos = ctrl.step(velocity=velocity)

            # ---- Send to robot ----
            robot_send_foot_positions(robot, foot_pos)

            # ---- Read state ----
            state = robot_get_state(robot)

            # ---- Update metrics ----
            metrics.update(
                vx=state["vx"],
                vz=state["vz"],
                roll=state["roll"],
                pitch=state["pitch"],
                body_height=state["body_height"],
            )

            # ---- Maintain 50 Hz ----
            sleep_s = max(0.0, DT - (time.time() - loop_t0))
            time.sleep(sleep_s)
            t += DT

    except KeyboardInterrupt:
        print("\n[Run] Ctrl-C — ending trial early.")

    result = metrics.end_trial()

    if robot is not None:
        robot_stop(robot)

    return result


# ===========================================================================
# Multi-trial condition runner
# ===========================================================================

def run_condition(mode: str,
                  disabled_leg_name: str,
                  n_trials: int = 5,
                  velocity: float = 0.20,
                  trial_duration: float = 10.0) -> dict:
    """Run n_trials of one condition and return aggregate stats."""
    logger = MetricsLogger(vx_cmd=velocity, trial_duration=trial_duration)

    for k in range(n_trials):
        print(f"\n[Condition {disabled_leg_name}]  Trial {k+1}/{n_trials}")
        result = run_trial(mode=mode, disabled_leg_name=disabled_leg_name,
                           velocity=velocity, trial_duration=trial_duration)
        logger.trial_results.append(result)

    return {
        "label":      disabled_leg_name,
        "J_mean":     logger.mean_J(),
        "J_std":      logger.std_J(),
        "S":          logger.survival_rate(),
        "n_trials":   n_trials,
    }


def run_all_conditions(mode: str = "smoke",
                       velocity: float = 0.20,
                       n_trials: int = 5):
    """
    Run all three experimental conditions and print the comparison table.
    Conditions match the proposal:
      (a) 4-leg trot, no fault
      (b) 3-leg gait, lift FR
      (c) 3-leg gait, lift HL
    """
    conditions = [
        ("NONE", "4-leg trot (baseline)"),
        ("FR",   "3-leg gait, lift FR"),
        ("HL",   "3-leg gait, lift HL"),
    ]

    agg_results = []
    for leg_name, description in conditions:
        print(f"\n{'═'*55}")
        print(f"  CONDITION: {description}")
        print(f"{'═'*55}")
        stats = run_condition(mode, leg_name, n_trials=n_trials,
                              velocity=velocity)
        stats["description"] = description
        agg_results.append(stats)

    # ---- Print summary table ----
    print(f"\n\n{'═'*65}")
    print(f"  FINAL RESULTS SUMMARY  ({n_trials} trials/condition)")
    print(f"{'═'*65}")
    print(f"  {'Condition':<35}  {'J mean ± std':<18}  {'S':>5}")
    print(f"  {'─'*35}  {'─'*18}  {'─'*5}")
    for r in agg_results:
        j_str = f"{r['J_mean']:.4f} ± {r['J_std']:.4f}"
        print(f"  {r['description']:<35}  {j_str:<18}  {r['S']:>4.2f}")
    print(f"{'═'*65}\n")

    # Save raw numbers
    _save_summary_csv(agg_results)
    return agg_results


def _save_summary_csv(results: list, path: str = "experiment_summary.csv"):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "description",
                                               "J_mean", "J_std", "S", "n_trials"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[Results] Summary saved → {path}")


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fault-tolerant three-leg locomotion experiment runner")
    parser.add_argument("--mode", default="smoke",
                        choices=["smoke", "sim", "real"],
                        help="smoke=no robot, sim=simulation, real=hardware")
    parser.add_argument("--disable-leg", default="FR",
                        choices=["NONE", "FR", "FL", "HR", "HL"],
                        help="Which leg to disable (NONE = 4-leg baseline)")
    parser.add_argument("--velocity", type=float, default=0.2,
                        help="Forward velocity command (m/s)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Trial duration (s)")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per condition")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run all 3 conditions × N trials")
    args = parser.parse_args()

    if args.all_conditions:
        run_all_conditions(mode=args.mode, velocity=args.velocity,
                           n_trials=args.trials)
    else:
        result = run_trial(
            mode=args.mode,
            disabled_leg_name=args.disable_leg,
            velocity=args.velocity,
            trial_duration=args.duration,
        )
        print(f"\nJ = {result['J']:.4f}   survived = {result['survived']}")
