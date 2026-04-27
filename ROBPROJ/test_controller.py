"""
runs the controller for 15s in smoke-test mode.
prints foot positions and verifies the transition.
"""

import numpy as np
import time
from three_leg_gait import ThreeLegGaitController, FR, FL, HR, HL, LEG_NAMES

def test_four_leg_trot():
    print("\n TEST 1: Four-leg trot for 3 seconds")
    ctrl = ThreeLegGaitController(dt=0.02)
    assert ctrl.mode == "four_leg"

    for step in range(150):  # 3s at 50Hz
        fp = ctrl.step(velocity=0.2)
        assert fp.shape == (4, 3), "foot positions must be (4,3)"
        # All feet should be at or below default height
        for leg in range(4):
            assert fp[leg, 2] <= ctrl.default_foot_pos[leg, 2] + ctrl.SWING_HEIGHT + 0.01, \
                f"Leg {leg} z={fp[leg,2]:.4f} exceeds swing envelope"

    print("  Four-leg trot OK")


def test_com_shifts():
    from three_leg_gait import COM_SHIFTS, DEFAULT_FOOT_POS, LEG_NAMES

    print("\ TEST 2: CoM shift values ")
    for leg in [FR, FL, HR, HL]:
        shift = COM_SHIFTS[leg]
        print(f"  Lift {LEG_NAMES[leg]}: shift = [{shift[0]:+.4f}, {shift[1]:+.4f}] m")
        # shift should be non-zero and reasonable (< 5 cm)
        assert 0 < np.linalg.norm(shift) < 0.05, f"Shift for {LEG_NAMES[leg]} out of range"
    print(" CoM shifts OK")


def test_three_leg_transition(leg_to_disable=FR):
    print(f"\n TEST 3: Transition + 3-leg walk (disable {LEG_NAMES[leg_to_disable]}) ")
    ctrl = ThreeLegGaitController(dt=0.02)
    DT   = 0.02

    for _ in range(150): # Run 4-leg for 3s
        ctrl.step(velocity=0.2)

    ctrl.disable_leg(leg_to_disable) # disable leg
    assert ctrl.mode == "three_leg"

    transition_done = False  # Run through transition + 5s more
    for step in range(400):
        fp = ctrl.step(velocity=0.2)
        t  = step * DT

        # Disabled leg must always be lifted after ramp completes)
        if not ctrl.is_transitioning:
            transition_done = True
            disabled_z = fp[leg_to_disable, 2]
            default_z  = ctrl.default_foot_pos[leg_to_disable, 2]
            assert disabled_z > default_z + 0.05, \
                f"Disabled leg z={disabled_z:.4f} not lifted (default={default_z:.4f})"

        # active legs should move within workspace
        for leg in range(4):
            if leg == leg_to_disable:
                continue
            assert fp[leg, 2] >= ctrl.default_foot_pos[leg, 2] - 0.02, \
                f"Leg {leg} z={fp[leg,2]:.4f} below floor at step {step}"

    assert transition_done, "Transition never completed"
    print(f" Three-leg walk OK (disable {LEG_NAMES[leg_to_disable]})")


def test_all_disabled_legs():
    print("\n TEST 4: All four legs can be disabled ")
    for leg in [FR, FL, HR, HL]:
        ctrl = ThreeLegGaitController(dt=0.02)
        ctrl.disable_leg(leg)
        for _ in range(10):
            fp = ctrl.step(velocity=0.2)
        print(f" Disable {LEG_NAMES[leg]} OK")


def demo_print():
    print("\n DEMO: Watch foot-z during FR disable (5s) ")
    print(f"{'t(s)':>5}  {'FR_z':>7}  {'FL_z':>7}  {'HR_z':>7}  {'HL_z':>7}  {'mode':<10}")
    print("─" * 55)

    ctrl = ThreeLegGaitController(dt=0.02)
    disable_sent = False

    for step in range(250):
        t = step * 0.02

        if t >= 1.0 and not disable_sent:
            ctrl.disable_leg(FR)
            disable_sent = True

        fp = ctrl.step(velocity=0.2)

        if step % 25 == 0:
            print(f"{t:5.1f}  "
                  f"{fp[0,2]:7.3f}  {fp[1,2]:7.3f}  "
                  f"{fp[2,2]:7.3f}  {fp[3,2]:7.3f}  "
                  f"{ctrl.mode}")

    print("\n  FR_z should increase to ~-0.04 after the 3s ramp.")


if __name__ == "__main__":
    test_four_leg_trot()
    test_com_shifts()
    for leg in [FR, FL, HR, HL]:
        test_three_leg_transition(leg)
    test_all_disabled_legs()
    demo_print()
    print("\n   All tests passed.\n")
