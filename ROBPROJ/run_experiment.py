"""
Experiment runner for the fault-tolerant three-legged locomotion project.

Lab 3 architecture:
  - SUBSCRIBES /joint_states                          (sensor_msgs/JointState)
  - PUBLISHES  /forward_command_controller/commands   (std_msgs/Float64MultiArray)
  - Per-leg FK + per-leg IK (scipy.optimize, L-BFGS-B)

This file:
  - Creates a ROS 2 node `Lab3Bridge` that uses the SAME topics + the SAME
    FK/IK math from Lab 3 (copied verbatim so behaviour is identical).
  - Runs the rclpy executor in a background thread so the main experiment
    loop stays a simple synchronous 50 Hz loop.
  - Optionally subscribes to /imu/data and /odom for state feedback.  If
    neither is available we estimate body_height proprioceptively and
    leave roll/pitch = 0  (a warning at start).

Three conditions (per proposal sec.IV):
  (a) baseline_4leg  - normal 4-leg trot, no fault
  (b) three_leg_FR   - 3-leg gait, FR disabled at t=3s
  (c) three_leg_HL   - 3-leg gait, HL disabled at t=3s
  5 trials per condition, 10 s each.

Usage
-----
# Quick smoke test (no robot, no ROS, no Pupper):
python3 run_experiment.py --mode smoke

# Full sim or real robot (requires ROS 2 + Pupper stack running):
python3 run_experiment.py --mode real --disable-leg FR
python3 run_experiment.py --mode real --all-conditions

Run order:
  1. Launch Lab 3 / Pupper bringup as you normally do.
  2. STOP lab_3 inverse_kinematics node (this script replaces it).
  3. Run script.
"""

import argparse
import threading
import time
from typing import Optional

import numpy as np
import scipy.optimize

from three_leg_gait import ThreeLegGaitController, FR, FL, HR, HL, LEG_NAMES
from metrics import MetricsLogger

LEG_MAP = {"FR": FR, "FL": FL, "HR": HR, "HL": HL, "NONE": None}
DT = 0.02   # 50 Hz control rate

JOINT_NAMES = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1",  "leg_back_r_2",  "leg_back_r_3",
    "leg_back_l_1",  "leg_back_l_2",  "leg_back_l_3",
]

# FK from lab 3

def _rx(a):  return np.array([[1,0,0,0],[0,np.cos(a),-np.sin(a),0],[0,np.sin(a),np.cos(a),0],[0,0,0,1]])
def _ry(a):  return np.array([[np.cos(a),0,np.sin(a),0],[0,1,0,0],[-np.sin(a),0,np.cos(a),0],[0,0,0,1]])
def _rz(a):  return np.array([[np.cos(a),-np.sin(a),0,0],[np.sin(a),np.cos(a),0,0],[0,0,1,0],[0,0,0,1]])
def _tr(x,y,z): return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])

def fr_leg_fk(theta):
    T01 = _tr(0.07500, -0.08350, 0) @ _rx(1.57080) @ _rz(theta[0])
    T12 = _ry(-1.57080) @ _rz(theta[1])
    T23 = _tr(0, -0.04940, 0.06850) @ _ry(1.57080) @ _rz(theta[2])
    T3e = _tr(0.06231, -0.06216, 0.01800)
    return (T01 @ T12 @ T23 @ T3e)[:3, 3]

def fl_leg_fk(theta):
    T01 = _tr(0.07500, 0.08350, 0) @ _rx(1.57080) @ _rz(-theta[0])
    T12 = _ry(-1.57080) @ _rz(theta[1])
    T23 = _tr(0, -0.04940, 0.06850) @ _ry(1.57080) @ _rz(-theta[2])
    T3e = _tr(0.06231, -0.06216, -0.01800)
    return (T01 @ T12 @ T23 @ T3e)[:3, 3]

def br_leg_fk(theta):   # = HR (hind right)
    T01 = _tr(-0.07500, -0.07250, 0) @ _rx(1.57080) @ _rz(theta[0])
    T12 = _ry(-1.57080) @ _rz(theta[1])
    T23 = _tr(0, -0.04940, 0.06850) @ _ry(1.57080) @ _rz(theta[2])
    T3e = _tr(0.06231, -0.06216, 0.01800)
    return (T01 @ T12 @ T23 @ T3e)[:3, 3]

def bl_leg_fk(theta):   # = HL (hind left)
    T01 = _tr(-0.07500, 0.07250, 0) @ _rx(1.57080) @ _rz(-theta[0])
    T12 = _ry(-1.57080) @ _rz(theta[1])
    T23 = _tr(0, -0.04940, 0.06850) @ _ry(1.57080) @ _rz(-theta[2])
    T3e = _tr(0.06231, -0.06216, -0.01800)
    return (T01 @ T12 @ T23 @ T3e)[:3, 3]

# order: FR, FL, HR(BR), HL(BL) ( matches three_leg_gait.py leg indices
FK_FUNCS = [fr_leg_fk, fl_leg_fk, br_leg_fk, bl_leg_fk]


def _ik_single_leg(target_ee, leg_index, x0):

    fk = FK_FUNCS[leg_index]
    target = np.asarray(target_ee, dtype=float)
    res = scipy.optimize.minimize(
        fun=lambda th: np.linalg.norm(fk(th) - target),
        x0=np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * 3,
        # lower maxiter than lab 3 cache (500)-> converges fast
        options={"ftol": 1e-9, "gtol": 1e-7, "maxiter": 60},
    )
    return res.x

def _import_ros2():
    """Lazy import so smoke mode works without ROS 2 installed."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Imu
    from std_msgs.msg import Float64MultiArray
    try:
        from nav_msgs.msg import Odometry
    except ImportError:
        Odometry = None
    return rclpy, Node, JointState, Imu, Float64MultiArray, Odometry


def _make_bridge_class():
    rclpy, Node, JointState, Imu, Float64MultiArray, Odometry = _import_ros2()

    class Lab3Bridge(Node):
        def __init__(self):
            super().__init__("three_leg_gait_bridge")

            # state caches (filled by callbacks)
            self._joint_positions: Optional[np.ndarray] = None
            self._imu_rpy: Optional[np.ndarray] = None
            self._odom_vx: Optional[float] = None
            self._odom_vz: Optional[float] = None
            self._odom_z:  Optional[float] = None

            # guesses for IK no1
            self._ik_warm = np.zeros((4, 3))
            self._last_cmd_vx = 0.0
            
            self.create_subscription(JointState, "joint_states", # Subscriptions
                                     self._on_joint_state, 10)
            for topic in ("imu/data", "imu", "imu_plugin/out", "imu/data_raw"):
                self.create_subscription(Imu, topic, self._on_imu, 10)
            if Odometry is not None:
                for topic in ("odom", "pupper/odom"):
                    self.create_subscription(Odometry, topic, self._on_odom, 10)

            # publisher
            self._cmd_pub = self.create_publisher(
                Float64MultiArray,
                "/forward_command_controller/commands",
                10,
            )

            self.get_logger().info("Lab3Bridge ready.")

        # callbacks
        def _on_joint_state(self, msg):
            try:
                idxs = [msg.name.index(n) for n in JOINT_NAMES]
            except ValueError:
                return
            self._joint_positions = np.array([msg.position[i] for i in idxs])

        def _on_imu(self, msg):
            qx, qy, qz, qw = (msg.orientation.x, msg.orientation.y,
                              msg.orientation.z, msg.orientation.w)
            sinr = 2.0 * (qw * qx + qy * qz)
            cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr, cosr)
            sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
            pitch = np.arcsin(sinp)
            siny = 2.0 * (qw * qz + qx * qy)
            cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny, cosy)
            self._imu_rpy = np.array([roll, pitch, yaw])

        def _on_odom(self, msg):
            self._odom_vx = float(msg.twist.twist.linear.x)
            self._odom_vz = float(msg.twist.twist.linear.z)
            self._odom_z  = float(msg.pose.pose.position.z)

        # api used 
        def wait_for_joint_state(self, timeout=5.0):
            t0 = time.time()
            while self._joint_positions is None:
                if time.time() - t0 > timeout:
                    return False
                time.sleep(0.05)
            return True

        def get_state_dict(self):
            # body_height: prefer odom; else proprioceptive estimate
            if self._odom_z is not None:
                body_height = self._odom_z
            elif self._joint_positions is not None:
                feet_z = [FK_FUNCS[i](self._joint_positions[3*i:3*i+3])[2]
                          for i in range(4)]
                body_height = float(-np.mean(feet_z))
            else:
                body_height = 0.14

            roll, pitch = (0.0, 0.0)
            if self._imu_rpy is not None:
                roll  = float(self._imu_rpy[0])
                pitch = float(self._imu_rpy[1])

            if self._odom_vx is not None:
                vx = self._odom_vx
                vz = self._odom_vz if self._odom_vz is not None else 0.0
            else:
                vx = self._last_cmd_vx
                vz = 0.0

            return {"vx": vx, "vz": vz, "roll": roll, "pitch": pitch,
                    "body_height": body_height}

        def send_foot_positions(self, foot_pos, cmd_vx=0.0):
            self._last_cmd_vx = cmd_vx
            joint_angles = np.zeros(12)
            for i in range(4):
                theta = _ik_single_leg(foot_pos[i], i, self._ik_warm[i])
                self._ik_warm[i] = theta
                joint_angles[3*i:3*i+3] = theta
            msg = Float64MultiArray()
            msg.data = joint_angles.tolist()
            self._cmd_pub.publish(msg)

        def send_zero_command(self):
            msg = Float64MultiArray()
            msg.data = [0.0] * 12
            self._cmd_pub.publish(msg)

        def has_imu(self):  return self._imu_rpy is not None
        def has_odom(self): return self._odom_vx is not None

    return Lab3Bridge, rclpy


_BRIDGE_STATE = {"node": None, "executor": None, "thread": None, "rclpy": None}

def robot_init(mode="real"):
    if mode == "smoke":
        print("[Robot] smoke mode - no ROS 2, using stubbed state.")
        return None

    Lab3Bridge, rclpy = _make_bridge_class()
    rclpy.init()
    node = Lab3Bridge()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    _BRIDGE_STATE.update(node=node, executor=executor, thread=thread, rclpy=rclpy)

    print("[Robot] Waiting for /joint_states ...")
    if not node.wait_for_joint_state(timeout=5.0):
        print("[Robot] WARNING: no /joint_states received in 5s - "
              "is your Pupper bringup running?")
    else:
        print("[Robot] /joint_states OK.")

    time.sleep(0.5)
    if not node.has_imu():
        print("[Robot] WARNING: no IMU on /imu, /imu/data, or /imu_plugin/out.")
        print("[Robot]   -> roll/pitch reported as 0; tilt-fall NOT detected.")
    if not node.has_odom():
        print("[Robot] NOTE: no /odom topic - vx will be reported as commanded value.")

    return node


def robot_get_state(robot):
    if robot is None:
        return {
            "vx":          0.18 + np.random.normal(0, 0.02),
            "vz":          np.random.normal(0, 0.01),
            "roll":        np.random.normal(0, 0.04),
            "pitch":       np.random.normal(0, 0.04),
            "body_height": 0.135 + np.random.normal(0, 0.003),
        }
    return robot.get_state_dict()


def robot_send_foot_positions(robot, foot_pos, cmd_vx=0.0):
    if robot is None:
        return
    robot.send_foot_positions(foot_pos, cmd_vx=cmd_vx)


def robot_stop(robot):
    if robot is None:
        return
    try:
        robot.send_zero_command()
        time.sleep(0.2)
    except Exception as e:
        print(f"[Robot] WARNING during stop: {e}")
    rclpy = _BRIDGE_STATE["rclpy"]
    executor = _BRIDGE_STATE["executor"]
    if executor is not None:
        executor.shutdown()
    if _BRIDGE_STATE["node"] is not None:
        _BRIDGE_STATE["node"].destroy_node()
    if rclpy is not None and rclpy.ok():
        rclpy.shutdown()
    _BRIDGE_STATE.update(node=None, executor=None, thread=None, rclpy=None)


#trial 
def run_trial(mode, disabled_leg_name, velocity=0.20,
              trial_duration=10.0, disable_at_t=3.0):
    disabled_leg = LEG_MAP[disabled_leg_name.upper()]
    is_fault_trial = disabled_leg is not None
    label = ("4-leg baseline" if not is_fault_trial
             else f"3-leg [{disabled_leg_name}]")
    print(f"\n{'-'*55}")
    print(f"  TRIAL  |  {label}  |  {mode} mode")
    print(f"{'-'*55}")

    if mode == "real":
        print("  Position robot on flat surface.  Press Enter to start...")
        input()

    ctrl    = ThreeLegGaitController(dt=DT)
    metrics = MetricsLogger(vx_cmd=velocity, trial_duration=trial_duration)
    robot   = robot_init(mode=mode) if mode in ("sim", "real") else None

    metrics.start_trial()
    t = 0.0
    leg_disabled = False

    try:
        while t < trial_duration:
            loop_t0 = time.time()

            if is_fault_trial and not leg_disabled and t >= disable_at_t:
                ctrl.disable_leg(disabled_leg)
                leg_disabled = True

            foot_pos = ctrl.step(velocity=velocity)
            robot_send_foot_positions(robot, foot_pos, cmd_vx=velocity)
            state = robot_get_state(robot)

            metrics.update(
                vx=state["vx"], vz=state["vz"],
                roll=state["roll"], pitch=state["pitch"],
                body_height=state["body_height"],
            )

            sleep_s = max(0.0, DT - (time.time() - loop_t0))
            time.sleep(sleep_s)
            t += DT
    except KeyboardInterrupt:
        print("\n[Run] Ctrl-C - ending trial early.")

    result = metrics.end_trial()
    if robot is not None:
        robot_stop(robot)
    return result


def run_condition(mode, disabled_leg_name, n_trials=5,
                  velocity=0.20, trial_duration=10.0):
    logger = MetricsLogger(vx_cmd=velocity, trial_duration=trial_duration)
    for k in range(n_trials):
        print(f"\n[Condition {disabled_leg_name}]  Trial {k+1}/{n_trials}")
        result = run_trial(mode=mode, disabled_leg_name=disabled_leg_name,
                           velocity=velocity, trial_duration=trial_duration)
        logger.trial_results.append(result)
    return {
        "label":    disabled_leg_name,
        "J_mean":   logger.mean_J(),
        "J_std":    logger.std_J(),
        "S":        logger.survival_rate(),
        "n_trials": n_trials,
    }


def run_all_conditions(mode="smoke", velocity=0.20, n_trials=5):
    conditions = [
        ("NONE", "4-leg trot (baseline)"),
        ("FR",   "3-leg gait, lift FR"),
        ("HL",   "3-leg gait, lift HL"),
    ]
    agg_results = []
    for leg_name, description in conditions:
        print(f"\n{'='*55}")
        print(f"  CONDITION: {description}")
        print(f"{'='*55}")
        stats = run_condition(mode, leg_name, n_trials=n_trials,
                              velocity=velocity)
        stats["description"] = description
        agg_results.append(stats)

    print(f"\n\n{'='*65}")
    print(f"  FINAL RESULTS SUMMARY  ({n_trials} trials/condition)")
    print(f"{'='*65}")
    print(f"  {'Condition':<35}  {'J mean +/- std':<18}  {'S':>5}")
    print(f"  {'-'*35}  {'-'*18}  {'-'*5}")
    for r in agg_results:
        j_str = f"{r['J_mean']:.4f} +/- {r['J_std']:.4f}"
        print(f"  {r['description']:<35}  {j_str:<18}  {r['S']:>4.2f}")
    print(f"{'='*65}\n")

    _save_summary_csv(agg_results)
    return agg_results


def _save_summary_csv(results, path="experiment_summary.csv"):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "description",
                                               "J_mean", "J_std", "S", "n_trials"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[Results] Summary saved -> {path}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fault-tolerant three-leg locomotion experiment runner")
    parser.add_argument("--mode", default="smoke",
                        choices=["smoke", "sim", "real"])
    parser.add_argument("--disable-leg", default="FR",
                        choices=["NONE", "FR", "FL", "HR", "HL"])
    parser.add_argument("--velocity", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--all-conditions", action="store_true")
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
