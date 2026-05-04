#!/usr/bin/env python3
"""
apple_pick_place.py
-------------------
Subscreve /apple_pose (frame 'base') e executa pick & place da maçã
com o SO-ARM101 via MoveIt 2.

Parâmetros (ajustar com --ros-args -p <param>:=<valor>):
  z_approach          – altura de aproximação acima da maçã (m)
  z_pick              – altura de descida até à maçã (m)   ← calibrar
  z_carry             – altura de transporte após pick (m)
  drop_x/y/z          – posição de largada no frame base
  tcp_offset_radial   – avanço radial do TCP em direcção à maçã (m)
  jaw_side_offset     – deslocamento lateral para compensar jaw fixo (m)
  jaw_open            – abertura máxima da garra (rad)
  jaw_preclose        – pré-fecho antes de descer (rad)
  jaw_close           – fecho no pick (rad)
  gripper_vel_scale   – velocidade da garra (0–1)
  ori_x/y/z/w         – quaternião da garra durante o pick
  wrist_pitch_offset  – offset de calibração do Wrist_Pitch (rad)
  wrist_roll_rad      – ângulo fixo do Wrist_Roll (rad)
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_configs_utils import MoveItConfigsBuilder


def plan_and_execute(robot, planning_component, logger, group_name,
                     plan_params=None, sleep_time=0.0):
    plan_result = (planning_component.plan(parameters=plan_params)
                   if plan_params else planning_component.plan())
    if plan_result:
        try:
            robot.execute(group_name, plan_result.trajectory, blocking=True)
        except Exception as exc:
            logger.error(f"Falha ao executar: {exc}")
            return False
    else:
        logger.error("Planeamento falhou!")
        return False
    time.sleep(sleep_time)
    return True


class ApplePickPlace(Node):

    def __init__(self):
        super().__init__('apple_pick_place')

        self.declare_parameter('base_frame',         'base')
        self.declare_parameter('tcp_offset_radial',  -0.03)
        self.declare_parameter('jaw_side_offset',     0.02)
        self.declare_parameter('z_approach',          0.20)
        self.declare_parameter('z_pick',              0.13)
        self.declare_parameter('z_carry',             0.25)
        self.declare_parameter('drop_x',              0.20)
        self.declare_parameter('drop_y',             -0.15)
        self.declare_parameter('drop_z',              0.30)
        self.declare_parameter('ori_x',              -0.024)
        self.declare_parameter('ori_y',              -0.024)
        self.declare_parameter('ori_z',              -0.710)
        self.declare_parameter('ori_w',               0.704)
        self.declare_parameter('wrist_roll_rad',     -math.pi / 2)
        self.declare_parameter('wrist_pitch_offset',  0.057)
        self.declare_parameter('grasp_jaw_threshold', 0.05)
        self.declare_parameter('max_pick_retries',    2)
        self.declare_parameter('vel_scale',           0.7)
        self.declare_parameter('gripper_vel_scale',   0.2)
        self.declare_parameter('jaw_open',            1.4483)
        self.declare_parameter('jaw_preclose',        0.55)
        self.declare_parameter('jaw_close',           0.3)

        moveit_config = (
            MoveItConfigsBuilder("so101_new_calib",
                                 package_name="so_arm_moveit_config")
            .to_moveit_configs()
        )
        config_dict = moveit_config.to_dict()
        pipelines   = config_dict.pop("planning_pipelines")
        config_dict["planning_pipelines"] = {"pipeline_names": pipelines}

        self.robot       = MoveItPy(node_name="moveit_py_apple_pick",
                                    config_dict=config_dict)
        self.arm         = self.robot.get_planning_component("arm")
        self.gripper_arm = self.robot.get_planning_component("gripper")
        self.robot_model = self.robot.get_robot_model()
        self.logger      = self.get_logger()

        self.arm_params = PlanRequestParameters(self.robot)
        self.arm_params.planning_pipeline = "ompl"
        self.arm_params.planner_id        = "RRTConnect"
        self.arm_params.planning_time     = 5.0
        self.arm_params.planning_attempts = 10

        self.gripper_params = PlanRequestParameters(self.robot)
        self.gripper_params.planning_pipeline = "ompl"
        self.gripper_params.planner_id        = "RRTConnect"
        self.gripper_params.planning_time     = 3.0

        self._lock = threading.Lock()
        self.create_subscription(Pose, '/apple_pose', self._on_apple_pose, 10)
        self.logger.info("ApplePickPlace ativo — a ouvir /apple_pose...")

    def _current_joints(self) -> dict:
        with self.robot.get_planning_scene_monitor().read_only() as scene:
            return dict(scene.current_state.joint_positions)

    def _gripper_grasped(self) -> bool:
        threshold = float(self.get_parameter('grasp_jaw_threshold').value)
        jaw = self._current_joints().get("Jaw", 0.0)
        self.logger.info(f"Jaw após fechar: {jaw:.4f} (limiar={threshold:.4f})")
        return jaw > threshold

    def _make_pose(self, x, y, z) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id    = self.get_parameter('base_frame').value
        pose.pose.position.x    = float(x)
        pose.pose.position.y    = float(y)
        pose.pose.position.z    = float(z)
        pose.pose.orientation.x = float(self.get_parameter('ori_x').value)
        pose.pose.orientation.y = float(self.get_parameter('ori_y').value)
        pose.pose.orientation.z = float(self.get_parameter('ori_z').value)
        pose.pose.orientation.w = float(self.get_parameter('ori_w').value)
        return pose

    def _move_arm(self, x, y, z, label="") -> bool:
        self.arm.set_start_state_to_current_state()
        state = RobotState(self.robot_model)
        if not state.set_from_ik("arm", self._make_pose(x, y, z).pose, "gripper", 5.0):
            self.logger.error(f"IK falhou {label}({x:.3f},{y:.3f},{z:.3f})")
            return False

        joints      = dict(state.joint_positions)
        pitch       = joints.get("Pitch", 0.0)
        elbow       = joints.get("Elbow", 0.0)
        wp_offset   = float(self.get_parameter('wrist_pitch_offset').value)
        joints["Wrist_Pitch"] = math.pi / 2.0 - (pitch + elbow) - wp_offset
        joints["Wrist_Roll"]  = float(self.get_parameter('wrist_roll_rad').value)
        state.joint_positions = joints

        self.logger.info(
            f"Punho: Pitch={math.degrees(pitch):.1f}° Elbow={math.degrees(elbow):.1f}°"
            f" → WristPitch={math.degrees(joints['Wrist_Pitch']):.1f}°"
        )

        try:
            self.arm.set_goal_state(robot_state=state)
        except TypeError:
            jmg = self.robot_model.get_joint_model_group("arm")
            jc  = construct_joint_constraint(robot_state=state, joint_model_group=jmg)
            self.arm.set_goal_state(motion_plan_constraints=[jc])

        self.arm_params.max_velocity_scaling_factor = float(
            self.get_parameter('vel_scale').value)
        self.logger.info(f"A mover {label}→ ({x:.3f},{y:.3f},{z:.3f})")
        return plan_and_execute(self.robot, self.arm, self.logger,
                                group_name="arm",
                                plan_params=self.arm_params,
                                sleep_time=0.5)

    def _set_gripper(self, opening, vel_scale: float | None = None) -> bool:
        self.gripper_arm.set_start_state_to_current_state()
        st = RobotState(self.robot_model)
        st.joint_positions = {"Jaw": float(opening)}
        jmg = self.robot_model.get_joint_model_group("gripper")
        jc  = construct_joint_constraint(robot_state=st, joint_model_group=jmg)
        self.gripper_arm.set_goal_state(motion_plan_constraints=[jc])
        if vel_scale is not None:
            self.gripper_params.max_velocity_scaling_factor = vel_scale
        return plan_and_execute(self.robot, self.gripper_arm, self.logger,
                                group_name="gripper",
                                plan_params=self.gripper_params,
                                sleep_time=0.3)

    def _on_apple_pose(self, msg: Pose):
        if not self._lock.acquire(blocking=False):
            self.logger.warn("Pick em curso — a ignorar nova deteção")
            return
        try:
            self._run_pick_place(msg)
        except Exception as exc:
            self.logger.error(f"Excepção no pick: {exc}")
        finally:
            self._lock.release()

    def _run_pick_place(self, msg: Pose):
        ax   = msg.position.x
        ay   = msg.position.y
        dist = math.sqrt(ax * ax + ay * ay)

        tcp_r    = float(self.get_parameter('tcp_offset_radial').value)
        side_off = float(self.get_parameter('jaw_side_offset').value)

        if dist > 1e-3:
            if tcp_r != 0.0:
                ax += tcp_r * (ax / dist)
                ay += tcp_r * (ay / dist)
            if side_off != 0.0:
                ax += side_off * (-ay / dist)
                ay += side_off * ( ax / dist)

        self.logger.info(
            f"Maçã em base: ({msg.position.x:.3f}, {msg.position.y:.3f})"
            f" → IK target: ({ax:.3f}, {ay:.3f})"
        )

        z_approach   = float(self.get_parameter('z_approach').value)
        z_pick       = float(self.get_parameter('z_pick').value)
        z_carry      = float(self.get_parameter('z_carry').value)
        drop_x       = float(self.get_parameter('drop_x').value)
        drop_y       = float(self.get_parameter('drop_y').value)
        drop_z       = float(self.get_parameter('drop_z').value)
        max_retries  = int(self.get_parameter('max_pick_retries').value)
        jaw_open     = float(self.get_parameter('jaw_open').value)
        jaw_preclose = float(self.get_parameter('jaw_preclose').value)
        jaw_close    = float(self.get_parameter('jaw_close').value)
        g_vel        = float(self.get_parameter('gripper_vel_scale').value)
        vel_orig     = float(self.get_parameter('vel_scale').value)

        self._set_gripper(jaw_open, vel_scale=g_vel)

        if not self._move_arm(ax, ay, z_approach, "aproximação"):
            return

        self._set_gripper(jaw_preclose, vel_scale=g_vel)

        grasped = False
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.warn(f"Retry {attempt}/{max_retries}")
                self._set_gripper(jaw_open, vel_scale=g_vel)
                if not self._move_arm(ax, ay, z_approach, "reposição"):
                    return
                self._set_gripper(jaw_preclose, vel_scale=g_vel)

            self.arm_params.max_velocity_scaling_factor = vel_orig * 0.5
            if not self._move_arm(ax, ay, z_pick, "pick"):
                self.arm_params.max_velocity_scaling_factor = vel_orig
                return
            self.arm_params.max_velocity_scaling_factor = vel_orig

            self._set_gripper(jaw_close, vel_scale=g_vel)
            if self._gripper_grasped():
                grasped = True
                self.logger.info("Maçã agarrada!")
                break
            self.logger.warn(f"Garra fechou em vazio (tentativa {attempt + 1})")

        if not grasped:
            self.logger.error("Não conseguiu agarrar a maçã — a abortar.")
            self._set_gripper(jaw_open, vel_scale=g_vel)
            return

        if not self._move_arm(ax, ay, z_carry, "levantar"):
            self._set_gripper(jaw_open, vel_scale=g_vel)
            return

        if not self._move_arm(drop_x, drop_y, drop_z, "drop"):
            self._set_gripper(jaw_open, vel_scale=g_vel)
            return

        self._set_gripper(jaw_open, vel_scale=g_vel)
        self.logger.info("Pick & place da maçã concluído!")


def main(args=None):
    rclpy.init(args=args)
    node = ApplePickPlace()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        thread.join()


if __name__ == '__main__':
    main()


import math
import threading
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_configs_utils import MoveItConfigsBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plan_and_execute(robot, planning_component, logger, group_name,
                     plan_params=None, sleep_time=0.0):
    plan_result = (planning_component.plan(parameters=plan_params)
                   if plan_params else planning_component.plan())
    if plan_result:
        try:
            robot.execute(group_name, plan_result.trajectory, blocking=True)
        except Exception as exc:
            logger.error(f"Falha ao executar: {exc}")
            return False
    else:
        logger.error("Planeamento falhou!")
        return False
    time.sleep(sleep_time)
    return True


# ---------------------------------------------------------------------------
# Controlador
# ---------------------------------------------------------------------------

class ApplePickPlace(Node):

    def __init__(self):
        super().__init__('apple_pick_place')

        self.declare_parameter('base_frame', 'base')

        # Offset TCP: compensa a distância gripper→jaw ao longo do eixo radial.
        # Positivo = avança em direcção à maçã.
        self.declare_parameter('tcp_offset_radial', -0.03)

        # Offset lateral da garra: a jaw direita é fixa, a esquerda é móvel.
        # Desloca o target IK perpendicularmente à direcção da maçã para que
        # a maçã fique centrada entre os dois lados da garra.
        # Positivo = desloca para a esquerda do robot (eixo +X perpendicular).
        # Ajustar em incrementos de 0.005 m até a maçã ficar centrada.
        self.declare_parameter('jaw_side_offset', 0.02)

        # ── Alturas fixas (calibrar com ros2 param set) ────────────────────
        self.declare_parameter('z_approach', 0.20)
        self.declare_parameter('z_pick',     0.13)   # CALIBRAR: descer até à maçã
        self.declare_parameter('z_carry',    0.25)

        # Destino de largada
        self.declare_parameter('drop_x',  0.20)
        self.declare_parameter('drop_y', -0.15)
        self.declare_parameter('drop_z',  0.30)

        # Orientação da garra durante pick
        self.declare_parameter('ori_x', -0.024)
        self.declare_parameter('ori_y', -0.024)
        self.declare_parameter('ori_z', -0.710)
        self.declare_parameter('ori_w',  0.704)

        # Wrist_Roll constante
        self.declare_parameter('wrist_roll_rad', -math.pi / 2)

        # Offset de calibração do Wrist_Pitch
        self.declare_parameter('wrist_pitch_offset', 0.057)

        # Verificação de agarre
        self.declare_parameter('grasp_jaw_threshold', 0.05)
        self.declare_parameter('max_pick_retries', 2)

        # Velocidade e garra
        self.declare_parameter('vel_scale',         0.7)
        self.declare_parameter('gripper_vel_scale', 0.2)   # garra lenta = menos impacto
        self.declare_parameter('jaw_open',          1.4483)
        # Pré-fecho: apertura ligeiramente maior que a maçã.
        # A garra desce já semi-fechada → contacto suave em vez de slam.
        # Ajustar: deve ser um pouco maior que o diâmetro da maçã em rad.
        self.declare_parameter('jaw_preclose',      0.55)
        self.declare_parameter('jaw_close',         0.3)

        # ---- MoveItPy --------------------------------------------------------
        moveit_config = (
            MoveItConfigsBuilder("so101_new_calib",
                                 package_name="so_arm_moveit_config")
            .to_moveit_configs()
        )
        config_dict = moveit_config.to_dict()
        pipelines   = config_dict.pop("planning_pipelines")
        config_dict["planning_pipelines"] = {"pipeline_names": pipelines}

        self.robot       = MoveItPy(node_name="moveit_py_apple_pick",
                                    config_dict=config_dict)
        self.arm         = self.robot.get_planning_component("arm")
        self.gripper_arm = self.robot.get_planning_component("gripper")
        self.robot_model = self.robot.get_robot_model()
        self.logger      = self.get_logger()

        self.arm_params = PlanRequestParameters(self.robot)
        self.arm_params.planning_pipeline  = "ompl"
        self.arm_params.planner_id         = "RRTConnect"
        self.arm_params.planning_time      = 5.0
        self.arm_params.planning_attempts  = 10

        self.gripper_params = PlanRequestParameters(self.robot)
        self.gripper_params.planning_pipeline = "ompl"
        self.gripper_params.planner_id        = "RRTConnect"
        self.gripper_params.planning_time     = 3.0

        self._lock = threading.Lock()

        # A pose já vem em frame 'base' do apple_detector
        self.create_subscription(Pose, '/apple_pose', self._on_apple_pose, 10)
        self.logger.info("ApplePickPlace ativo — a ouvir /apple_pose...")

    # ---- Utilidades ----------------------------------------------------------

    def _current_joints(self) -> dict:
        with self.robot.get_planning_scene_monitor().read_only() as scene:
            return dict(scene.current_state.joint_positions)

    def _gripper_grasped(self) -> bool:
        threshold = float(self.get_parameter('grasp_jaw_threshold').value)
        jaw = self._current_joints().get("Jaw", 0.0)
        self.logger.info(f"Jaw após fechar: {jaw:.4f} (limiar={threshold:.4f})")
        return jaw > threshold

    def _make_pose(self, x, y, z) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id    = self.get_parameter('base_frame').value
        pose.pose.position.x    = float(x)
        pose.pose.position.y    = float(y)
        pose.pose.position.z    = float(z)
        pose.pose.orientation.x = float(self.get_parameter('ori_x').value)
        pose.pose.orientation.y = float(self.get_parameter('ori_y').value)
        pose.pose.orientation.z = float(self.get_parameter('ori_z').value)
        pose.pose.orientation.w = float(self.get_parameter('ori_w').value)
        return pose

    def _move_arm(self, x, y, z, label="") -> bool:
        self.arm.set_start_state_to_current_state()
        pose  = self._make_pose(x, y, z)
        state = RobotState(self.robot_model)
        ok    = state.set_from_ik("arm", pose.pose, "gripper", 5.0)
        if not ok:
            self.logger.error(f"IK falhou {label}({x:.3f},{y:.3f},{z:.3f})")
            return False

        joints = dict(state.joint_positions)
        pitch_angle = joints.get("Pitch", 0.0)
        elbow_angle = joints.get("Elbow", 0.0)
        wp_offset   = float(self.get_parameter('wrist_pitch_offset').value)
        wrist_pitch = math.pi / 2.0 - (pitch_angle + elbow_angle) - wp_offset
        wrist_roll  = float(self.get_parameter('wrist_roll_rad').value)

        joints["Wrist_Pitch"] = wrist_pitch
        joints["Wrist_Roll"]  = wrist_roll

        self.logger.info(
            f"Punho: Pitch={math.degrees(pitch_angle):.1f}°"
            f" Elbow={math.degrees(elbow_angle):.1f}°"
            f" → WristPitch={math.degrees(wrist_pitch):.1f}°"
            f" WristRoll={math.degrees(wrist_roll):.1f}°"
        )
        state.joint_positions = joints

        try:
            self.arm.set_goal_state(robot_state=state)
        except TypeError:
            jmg = self.robot_model.get_joint_model_group("arm")
            jc  = construct_joint_constraint(robot_state=state,
                                             joint_model_group=jmg)
            self.arm.set_goal_state(motion_plan_constraints=[jc])

        self.arm_params.max_velocity_scaling_factor = float(
            self.get_parameter('vel_scale').value)
        self.logger.info(f"A mover {label}→ ({x:.3f},{y:.3f},{z:.3f})")
        return plan_and_execute(self.robot, self.arm, self.logger,
                                group_name="arm",
                                plan_params=self.arm_params,
                                sleep_time=0.5)

    def _set_gripper(self, opening, vel_scale: float | None = None) -> bool:
        self.gripper_arm.set_start_state_to_current_state()
        st = RobotState(self.robot_model)
        st.joint_positions = {"Jaw": float(opening)}
        jmg = self.robot_model.get_joint_model_group("gripper")
        jc  = construct_joint_constraint(robot_state=st, joint_model_group=jmg)
        self.gripper_arm.set_goal_state(motion_plan_constraints=[jc])
        if vel_scale is not None:
            self.gripper_params.max_velocity_scaling_factor = vel_scale
        return plan_and_execute(self.robot, self.gripper_arm, self.logger,
                                group_name="gripper",
                                plan_params=self.gripper_params,
                                sleep_time=0.3)

    # ---- Callback ------------------------------------------------------------

    def _on_apple_pose(self, msg: Pose):
        if not self._lock.acquire(blocking=False):
            self.logger.warn("Pick em curso — a ignorar nova deteção")
            return
        try:
            self._run_pick_place(msg)
        except Exception as exc:
            self.logger.error(f"Excepção no pick: {exc}")
        finally:
            self._lock.release()

    # ---- Sequência -----------------------------------------------------------

    def _run_pick_place(self, msg: Pose):
        # A pose já está em frame 'base' (publicada pelo apple_detector via TF2)
        ax = msg.position.x
        ay = msg.position.y

        # Offset radial opcional para compensar TCP→jaw
        tcp_r = float(self.get_parameter('tcp_offset_radial').value)
        dist  = math.sqrt(ax * ax + ay * ay)
        if tcp_r != 0.0 and dist > 1e-3:
            ax += tcp_r * (ax / dist)
            ay += tcp_r * (ay / dist)

        # Offset lateral: perpendicular à direcção da maçã, no plano XY.
        # Compensa o facto de a jaw direita ser fixa e a esquerda ser móvel —
        # a maçã precisa de estar ligeiramente para o lado do jaw fixo.
        side_off = float(self.get_parameter('jaw_side_offset').value)
        if side_off != 0.0 and dist > 1e-3:
            # Vector perpendicular (rotação 90° no plano XY): (-dy, dx) / dist
            perp_x = -ay / dist
            perp_y =  ax / dist
            ax += side_off * perp_x
            ay += side_off * perp_y

        self.logger.info(
            f"Maçã detetada em base: ({msg.position.x:.3f}, {msg.position.y:.3f})"
            f" → IK target: ({ax:.3f}, {ay:.3f})"
            f" [radial={tcp_r:.3f} lateral={side_off:.3f}]"
        )

        z_approach  = float(self.get_parameter('z_approach').value)
        z_pick      = float(self.get_parameter('z_pick').value)
        z_carry     = float(self.get_parameter('z_carry').value)
        drop_x      = float(self.get_parameter('drop_x').value)
        drop_y      = float(self.get_parameter('drop_y').value)
        drop_z      = float(self.get_parameter('drop_z').value)
        max_retries   = int(self.get_parameter('max_pick_retries').value)
        jaw_open      = float(self.get_parameter('jaw_open').value)
        jaw_preclose  = float(self.get_parameter('jaw_preclose').value)
        jaw_close     = float(self.get_parameter('jaw_close').value)
        g_vel         = float(self.get_parameter('gripper_vel_scale').value)

        # 1. Abrir garra completamente
        self._set_gripper(jaw_open, vel_scale=g_vel)

        # 2. Aproximação acima da maçã
        if not self._move_arm(ax, ay, z_approach, "aproximação"):
            return

        # 3. Pré-fechar: garra desce já semi-fechada para contacto suave
        self._set_gripper(jaw_preclose, vel_scale=g_vel)
        self.logger.info(f"Pré-fecho: jaw={jaw_preclose:.3f} (vel={g_vel:.2f})")

        # 4. Pick com retry
        grasped = False
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.warn(f"Retry {attempt}/{max_retries}")
                self._set_gripper(jaw_open, vel_scale=g_vel)
                if not self._move_arm(ax, ay, z_approach, "reposição"):
                    return
                self._set_gripper(jaw_preclose, vel_scale=g_vel)

            # Descer devagar até à maçã (garra já semi-fechada)
            vel_orig = float(self.get_parameter('vel_scale').value)
            self.arm_params.max_velocity_scaling_factor = vel_orig * 0.5
            if not self._move_arm(ax, ay, z_pick, "pick"):
                self.arm_params.max_velocity_scaling_factor = vel_orig
                return
            self.arm_params.max_velocity_scaling_factor = vel_orig

            # Fechar completamente de forma lenta
            self._set_gripper(jaw_close, vel_scale=g_vel)

            if self._gripper_grasped():
                grasped = True
                self.logger.info("Maçã agarrada!")
                break
            self.logger.warn(f"Garra fechou em vazio (tentativa {attempt + 1})")

        if not grasped:
            self.logger.error("Não conseguiu agarrar a maçã — a abortar.")
            self._set_gripper(jaw_open, vel_scale=g_vel)
            return

        # 5. Levantar
        if not self._move_arm(ax, ay, z_carry, "levantar"):
            self._set_gripper(jaw_open, vel_scale=g_vel)
            return

        # 6. Transportar até ao destino
        if not self._move_arm(drop_x, drop_y, drop_z, "drop"):
            self._set_gripper(jaw_open, vel_scale=g_vel)
            return

        # 7. Largar
        self._set_gripper(jaw_open, vel_scale=g_vel)
        self.logger.info("Pick & place da maçã concluído!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ApplePickPlace()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        thread.join()


if __name__ == '__main__':
    main()
