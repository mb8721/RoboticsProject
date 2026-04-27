"""
metrics.py
==========
Performance metrics for the fault-tolerant locomotion project.

Tracks:
  J  – composite stability cost (lower = better)
  S  – survival rate across M trials (higher = better)

J formula (matches proposal §III):
  J = (1/N) * Σ_t [ w1*(vx−v̄x)² + w2*roll² + w3*pitch² + w4*vz² ]

S formula:
  S = (# trials walking ≥ T seconds without falling) / M

Fall criteria (from proposal §III):
  body height < 0.10 m   OR   |roll| > 45°

Usage
-----
from metrics import MetricsLogger

logger = MetricsLogger(vx_cmd=0.2, trial_duration=10.0)

# In control loop:
logger.update(vx, vz, roll, pitch, body_height)

# At end of trial:
result = logger.end_trial()

# After M trials:
print(logger.summary())
logger.save_csv("results_FR.csv")
"""

import csv
import time
import numpy as np
from typing import Optional


class MetricsLogger:
    """Tracks stability cost J and survival rate S across trials."""

    # ---- Reward weights (tune to emphasize what matters) ----
    W_VX    = 2.0    # forward velocity tracking
    W_VZ    = 1.0    # vertical velocity (bouncing)
    W_ROLL  = 1.5    # roll stability
    W_PITCH = 1.5    # pitch stability

    # ---- Fall detection thresholds ----
    FALL_HEIGHT_M       = 0.10            # body height (m) below which = fallen
    FALL_ROLL_RAD       = np.radians(45)  # roll angle beyond which = fallen
    FALL_GRACE_S        = 1.5             # ignore falls during first N seconds
    FALL_SUSTAIN_S      = 0.30            # require N seconds of sustained "fallen" before declaring

    def __init__(self, vx_cmd: float = 0.2, trial_duration: float = 10.0):
        """
        Parameters
        ----------
        vx_cmd : float
            Commanded forward velocity (m/s) used in cost calculation.
        trial_duration : float
            Required survival time (s) to count as a successful trial.
        """
        self.vx_cmd         = vx_cmd
        self.trial_duration = trial_duration

        self._costs: list   = []
        self._fell: bool    = False
        self._fall_time: Optional[float] = None
        self._fallen_since: Optional[float] = None  # debounce timer
        self._start_time    = None

        # Telemetry for diagnostics (post-trial summary of state ranges)
        self._roll_samples: list = []
        self._pitch_samples: list = []
        self._height_samples: list = []

        self.trial_results: list = []   # list of dicts

    # ------------------------------------------------------------------
    # Trial management
    # ------------------------------------------------------------------

    def start_trial(self):
        """Call at the beginning of each new trial."""
        self._costs     = []
        self._fell      = False
        self._fall_time = None
        self._fallen_since = None
        self._start_time = time.time()
        self._roll_samples   = []
        self._pitch_samples  = []
        self._height_samples = []

    def update(self,
               vx: float, vz: float,
               roll: float, pitch: float,
               body_height: float):
        """
        Record one timestep.  Call every control cycle.

        Parameters
        ----------
        vx          : forward velocity (m/s)
        vz          : vertical velocity (m/s)
        roll        : body roll angle (rad)
        pitch       : body pitch angle (rad)
        body_height : body CoM height above ground (m)
        """
        if self._fell:
            return   # trial already failed; stop accumulating

        now = time.time()
        elapsed = now - (self._start_time or now)

        # Telemetry (always recorded, even during grace)
        self._roll_samples.append(roll)
        self._pitch_samples.append(pitch)
        self._height_samples.append(body_height)

        # ---- Fall detection (with grace period + sustained-state debounce) ----
        looks_fallen = (body_height < self.FALL_HEIGHT_M or
                        abs(roll) > self.FALL_ROLL_RAD)

        if elapsed < self.FALL_GRACE_S:
            # Inside grace window — never declare a fall
            self._fallen_since = None
        elif looks_fallen:
            # Start (or continue) the sustained-fallen timer
            if self._fallen_since is None:
                self._fallen_since = now
            elif (now - self._fallen_since) >= self.FALL_SUSTAIN_S:
                self._fell = True
                self._fall_time = elapsed
                print(f"[Metrics] *** FALL DETECTED at t={self._fall_time:.2f}s "
                      f"(roll={np.degrees(roll):+.1f}°, "
                      f"height={body_height:.3f}m) ***")
                return
        else:
            # reset the sustained timer
            self._fallen_since = None

        # ---- Instantaneous cost ----
        cost = (self.W_VX    * (vx - self.vx_cmd) ** 2 +
                self.W_VZ    * vz ** 2 +
                self.W_ROLL  * roll ** 2 +
                self.W_PITCH * pitch ** 2)
        self._costs.append(cost)

    def end_trial(self) -> dict:
        """
        Finalize the current trial.  Returns a result dict and appends
        it to self.trial_results.

        Returns
        -------
        dict with keys: J, survived, duration_s, fell, fall_time_s
        """
        duration = (time.time() - self._start_time) if self._start_time else 0.0
        survived = (not self._fell) and (duration >= self.trial_duration)

        J = float(np.mean(self._costs)) if self._costs else float("inf")

        result = dict(
            J            = round(J, 6),
            survived     = survived,
            duration_s   = round(duration, 2),
            fell         = self._fell,
            fall_time_s  = round(self._fall_time, 2) if self._fall_time else None,
            n_steps      = len(self._costs),
        )
        self.trial_results.append(result)

        status = "SURVIVED ✓" if survived else ("FELL ✗" if self._fell else "TIMEOUT ✗")
        print(f"[Metrics] Trial {len(self.trial_results)} ended  |  "
              f"J={J:.4f}  |  {status}  |  {duration:.1f}s")

        # Telemetry: range of roll, pitch, body_height during trial
        if self._roll_samples:
            r_deg = np.degrees(self._roll_samples)
            p_deg = np.degrees(self._pitch_samples)
            h_m   = np.array(self._height_samples)
            print(f"[Telemetry]  roll  min/max/|max| = "
                  f"{r_deg.min():+6.1f}° / {r_deg.max():+6.1f}° / {np.abs(r_deg).max():5.1f}°")
            print(f"[Telemetry]  pitch min/max/|max| = "
                  f"{p_deg.min():+6.1f}° / {p_deg.max():+6.1f}° / {np.abs(p_deg).max():5.1f}°")
            print(f"[Telemetry]  height min/max      = "
                  f"{h_m.min():.3f}m / {h_m.max():.3f}m")
        return result

    
    def survival_rate(self) -> float:
        """S = fraction of completed trials that survived."""
        if not self.trial_results:
            return 0.0
        return sum(1 for r in self.trial_results if r["survived"]) / len(self.trial_results)

    def mean_J(self) -> float:
        """Mean stability cost across completed trials."""
        vals = [r["J"] for r in self.trial_results if r["J"] != float("inf")]
        return float(np.mean(vals)) if vals else float("inf")

    def std_J(self) -> float:
        """Std dev of stability cost."""
        vals = [r["J"] for r in self.trial_results if r["J"] != float("inf")]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def summary(self) -> str:
        n  = len(self.trial_results)
        ns = sum(1 for r in self.trial_results if r["survived"])
        return (
            f"\n{'='*50}\n"
            f"  METRICS SUMMARY  ({n} trials)\n"
            f"{'='*50}\n"
            f"  Survival rate  S = {self.survival_rate():.2f}  ({ns}/{n})\n"
            f"  Stability cost J = {self.mean_J():.4f} ± {self.std_J():.4f}\n"
            f"{'='*50}\n"
        )

    def save_csv(self, filepath: str = "metrics_results.csv"):
        """Save all trial results to a CSV file."""
        if not self.trial_results:
            print("[Metrics] No results to save.")
            return
        fieldnames = list(self.trial_results[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial"] + fieldnames)
            writer.writeheader()
            for i, row in enumerate(self.trial_results, start=1):
                writer.writerow({"trial": i, **row})
        print(f"[Metrics] Saved {len(self.trial_results)} trials → {filepath}")
