"""

Fault-tolerant three-legged gait controller for Pupper (ROB-UY-2004).

Extends the Lab 3 trot controller with:
  1. Per-leg health flag
  2. Smooth 3-second leg-lift transition
  3. CoM compensation (shifts support feet so projected CoM stays
     inside the support triangle)
  4. Three-leg walk gait (one leg swings at a time — most stable)

Inspired by: Feng et al., Control Engineering Practice, vol. 165, 2025

Integration:
  Replace (or wrap) Lab 3 foot-position computation with
  ThreeLegGaitController.step().  The output is a (4,3) numpy array
  of foot positions in the body frame, ready for  IK function.

Coordinate convention (matches Stanford/NYU Pupper):
  x  = forward
  y  = left (positive) / right (negative)
  z  = up   (foot at rest ~ -0.14 m, i.e. 14 cm below body)

Leg indices:
  FR=0  FL=1  HR=2  HL=3
"""

import numpy as np

FR, FL, HR, HL = 0, 1, 2, 3
LEG_NAMES = {FR: "FR", FL: "FL", HR: "HR", HL: "HL"}

# Default foot positions in the body frame (meters).
#   rf_ee_offset = [ 0.06, -0.09, 0]    +  z_stance = -0.14
#   lf_ee_offset = [ 0.06,  0.09, 0]    +  z_stance = -0.14
#   rb_ee_offset = [-0.11, -0.09, 0]    +  z_stance = -0.14
#   lb_ee_offset = [-0.11,  0.09, 0]    +  z_stance = -0.14

DEFAULT_FOOT_POS = np.array([
    [ 0.06, -0.09, -0.14],   # FR (rf in Lab 3)
    [ 0.06,  0.09, -0.14],   # FL (lf in Lab 3)
    [-0.11, -0.09, -0.14],   # HR (rb in Lab 3)
    [-0.11,  0.09, -0.14],   # HL (lb in Lab 3)
], dtype=float)

# Pre-computed CoM compensation shifts (meters, XY plane).
#
# When leg i is lifted the support polygon becomes a triangle of the
# other 3 feet.  To keep the body CoM projected inside that triangle
# we shift all support foot targets by -centroid * gain, which is
# equivalent to leaning the body toward the centroid.
#
# Foot XY layout:
#   FR (+0.08, -0.065)  FL (+0.08, +0.065)
#   HR (-0.08, -0.065)  HL (-0.08, +0.065)
#
# Centroids (gain=1):
#   Lift FR: centroid(FL,HR,HL) = ((-0.08+0.08-0.08)/3, (0.065-0.065+0.065)/3)
#            = (-0.0267, +0.0217)  -> shift feet by +centroid to lean body there
#   Lift FL: centroid(FR,HR,HL) = (-0.0267, -0.0217)
#   Lift HR: centroid(FR,FL,HL) = (+0.0267, +0.0217)
#   Lift HL: centroid(FR,FL,HR) = (+0.0267, -0.0217)

_FOOT_XY = DEFAULT_FOOT_POS[:, :2]
_COM_SHIFT_GAIN = 0.70   # fraction of full centroid shift to apply (tune 0.5-0.8)

def _precompute_com_shifts():
    shifts = {}
    for leg in [FR, FL, HR, HL]:
        support = [i for i in range(4) if i != leg]
        centroid = np.mean(_FOOT_XY[support], axis=0)
        # shift support feet toward centroid so body CoM is above triangle
        shifts[leg] = centroid * _COM_SHIFT_GAIN
    return shifts

COM_SHIFTS = _precompute_com_shifts()


class ThreeLegGaitController:
    """
    Drop-in replacement for the Lab 3 trot controller with fault-tolerance.

    Quick-start
    -----------
    ctrl = ThreeLegGaitController(dt=0.02)

    # Normal trot (4-leg):
    foot_pos = ctrl.step(velocity=0.2)   # call every 20 ms

    # Simulate leg failure (e.g. FR breaks):
    ctrl.disable_leg(FR)

    # Controller now smoothly lifts FR and walks on 3 legs:
    foot_pos = ctrl.step(velocity=0.2)
    """

    # Gait parameters 
    SWING_HEIGHT    = 0.05   # m max foot lift during swing
    STEP_LENGTH     = 0.05   # m max step forward per cycle
    SWING_FRACTION  = 0.35   # fraction of cycle spent in swing
    CYCLE_HZ        = 1.2    # gait cycles per second

    # fault-transition parameters (from Feng et al. §3.3) 
    LIFT_HEIGHT   = 0.10   # m  how high to hold the disabled foot
    LIFT_DURATION = 3.0    # s  ramp time (avoids inertial shock)

    def __init__(self, dt: float = 0.02,
                 default_foot_pos: np.ndarray = None):
        """
        Parameters
        ----------
        dt : float
            Control timestep in seconds (default 0.02 → 50 Hz).
        default_foot_pos : ndarray (4,3), optional
            Override default standing foot positions.
        """
        self.dt = dt

        self.default_foot_pos = (
            DEFAULT_FOOT_POS.copy() if default_foot_pos is None
            else np.array(default_foot_pos, dtype=float)
        )

        self.disabled_leg     = None   # None = four-leg trot
        self.transition_t     = 0.0
        self.is_transitioning = False

        # Gait phases in [0,1) for each leg.
        # Four-leg trot: diagonal pairs share phase.
        #   FR(0) + HL(3) swing together  phase 0.0
        #   FL(1) + HR(2) swing together  phase 0.5
        self.phases = np.array([0.0, 0.5, 0.5, 0.0], dtype=float)

        # Foot positions at swing start (updated each stance  to swing transition)
        self._swing_start = self.default_foot_pos.copy()
        self._in_swing    = np.zeros(4, dtype=bool)

        # Last commanded positions
        self.foot_commands = self.default_foot_pos.copy()

        # Dynamic step length (set per call to step() based on velocity)
        self._step_length = self.STEP_LENGTH


    def disable_leg(self, leg_idx: int):
        """
        Signal a leg failure.  Initiates a smooth 3-second lift transition.

        Parameters
        ----------
        leg_idx : int
            One of FR(0), FL(1), HR(2), HL(3).
        """
        if leg_idx not in [FR, FL, HR, HL]:
            raise ValueError(f"Invalid leg_idx {leg_idx}. Use FR=0, FL=1, HR=2, HL=3.")

        print(f"\n[ThreeLeg] *** Leg {LEG_NAMES[leg_idx]} DISABLED ***")
        print(f"[ThreeLeg] Starting {self.LIFT_DURATION:.0f}s smooth lift transition.")

        self.disabled_leg     = leg_idx
        self.transition_t     = 0.0
        self.is_transitioning = True

        # Switch to three-leg walk: space remaining legs evenly
        active = [i for i in range(4) if i != leg_idx]
        for k, leg in enumerate(active):
            self.phases[leg] = k / 3.0   # 0.0, 0.333, 0.667

    def step(self, velocity: float = 0.2) -> np.ndarray:
        """
        Advance by one timestep and return commanded foot positions.

        Parameters
        ----------
        velocity : float
            Desired forward velocity (m/s).  Set to 0 to stand still.

        Returns
        -------
        foot_pos : ndarray (4,3)
            [x, y, z] for each leg in body frame (FR, FL, HR, HL).
        """
        #update transition timer 
        if self.is_transitioning:
            self.transition_t += self.dt
            if self.transition_t >= self.LIFT_DURATION:
                self.is_transitioning = False
                print(f"[ThreeLeg] Transition complete.  Running 3-leg walk.")

        # phase advance 
        moving  = velocity > 0.01
        d_phase = (self.CYCLE_HZ * self.dt) if moving else 0.0

        # velocity = step_length * cycle_hz  ->  step_length = velocity / cycle_hz
        if moving:
            self._step_length = float(np.clip(velocity / self.CYCLE_HZ, 0.03, 0.08))
        else:
            self._step_length = self.STEP_LENGTH

        # coM shift for 3-leg mode 
        com_xy = COM_SHIFTS[self.disabled_leg] if self.disabled_leg is not None else np.zeros(2)

        #  Compute foot targets 
        foot_cmd = np.zeros((4, 3))

        for leg in range(4):
            if leg == self.disabled_leg:
                foot_cmd[leg] = self._disabled_pos(leg)
                continue

            # Detect stance to swing transition (for updating swing_start)
            prev_phase = self.phases[leg]
            self.phases[leg] = (self.phases[leg] + d_phase) % 1.0
            cur_phase = self.phases[leg]

            # Crossed 0 to entering new swing
            if prev_phase > self.SWING_FRACTION and cur_phase <= self.SWING_FRACTION:
                # Capture position at end of stance as swing start
                self._swing_start[leg] = foot_cmd[leg] if np.any(foot_cmd[leg]) else \
                    self._stance_pos(leg, 1.0)

            p = self.phases[leg]

            if p < self.SWING_FRACTION:
                t = p / self.SWING_FRACTION              # 0 to 1 within swing
                foot_cmd[leg] = self._swing_pos(leg, t)
            else:
                t = (p - self.SWING_FRACTION) / (1.0 - self.SWING_FRACTION)  # 0 to 1within stance
                foot_cmd[leg] = self._stance_pos(leg, t)

            # Apply CoM compensation to support legs
            foot_cmd[leg][0] += com_xy[0]
            foot_cmd[leg][1] += com_xy[1]

        self.foot_commands = foot_cmd
        return foot_cmd.copy()

    @property
    def mode(self) -> str:
        """Current mode: 'four_leg' or 'three_leg'."""
        return "three_leg" if self.disabled_leg is not None else "four_leg"

  
    def _swing_pos(self, leg: int, t: float) -> np.ndarray:
        """
        Swing trajectory using cycloid-like x and half-cosine z lift.
        Directly implements Feng et al. Eq. (1).

        t : float in [0,1]  (0 = start of swing, 1 = end)
        """
        start   = self._swing_start[leg]
        default = self.default_foot_pos[leg]

        # X: cycloid curve for smooth acceleration/deceleration
        end_x = default[0] + self._step_length / 2.0
        theta = 2.0 * np.pi * t
        x = (end_x - start[0]) * (theta - np.sin(theta)) / (2.0 * np.pi) + start[0]

        # Y: stay centered
        y = (start[1] + default[1]) / 2.0

        # Z: half-cosine lift  (z is negative, += raises foot)
        z = self.SWING_HEIGHT * (1.0 - np.cos(theta)) / 2.0 + default[2]

        return np.array([x, y, z])

    def _stance_pos(self, leg: int, t: float) -> np.ndarray:
        """
        Stance trajectory: foot pushes backward linearly.
        Directly implements Feng et al. Eq. (2).

        t : float in [0,1]  (0 = start of stance, 1 = end)
        """
        d = self.default_foot_pos[leg]
        x = d[0] + self._step_length * (0.5 - t)   # +L/2 to  -L/2
        return np.array([x, d[1], d[2]])

    def _disabled_pos(self, leg: int) -> np.ndarray:
        """
        Hold the disabled leg lifted.
        Ramps smoothly over LIFT_DURATION seconds.
        """
        pos = self.default_foot_pos[leg].copy()
        if self.is_transitioning:
            ramp = min(self.transition_t / self.LIFT_DURATION, 1.0)
        else:
            ramp = 1.0
        pos[2] += ramp * self.LIFT_HEIGHT   # z is negativeto  lifts foot
        return pos
