"""
apple_detector.py
-----------------
Detetor de maçãs usando YOLOv8n (COCO class 47 = apple) + fallback HSV.

Parâmetros úteis em runtime:
  conf_threshold  – confiança mínima YOLO (default 0.15)
  debug_mode      – True → mostra TODAS as deteções YOLO (diagnóstico)
  use_hsv         – True → fallback HSV para maçãs sintéticas/renderizadas
  z_dist          – distância estimada câmara→objeto (metros)
"""

import math

import cv2
import numpy as np
import rclpy
import tf2_ros
import tf2_geometry_msgs  # regista suporte a PointStamped no TF2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image

# YOLOv8 via ultralytics
try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit(
        "Ultralytics não instalado. Corre: pip install ultralytics"
    ) from e

COCO_APPLE_CLASS = 47 #apple


class AppleDetector(Node):
    """Nó ROS 2 que deteta maçãs com YOLOv8 e publica a sua pose no clique do rato."""

    def __init__(self):
        super().__init__("apple_detector")

        # ── Parâmetros ──────────────────────────────────────────────────────
        self.declare_parameter("image_topic",      "/camera/image_raw")
        self.declare_parameter("apple_pose_topic", "/apple_pose")
        self.declare_parameter("yolo_model",       "yolov8n.pt")   # baixado automático
        self.declare_parameter("conf_threshold",   0.15)           # confiança mínima (baixo para sim)
        self.declare_parameter("z_dist",           0.8)            # distância estimada (m)
        self.declare_parameter("show_conf",        True)           # mostrar % confiança
        # debug_mode=True → mostra TODOS os objetos detetados (para diagnóstico)
        self.declare_parameter("debug_mode",       False)
        # use_hsv=True → fallback de cor HSV para maçãs sintéticas
        self.declare_parameter("use_hsv",          True)

        # Intrínsecos da câmara (pixels) — câmara 1280×720, fl=18.14 mm
        self.declare_parameter("fx", 1108.6)
        self.declare_parameter("fy", 1108.6)
        self.declare_parameter("cx",  640.0)
        self.declare_parameter("cy",  360.0)

        # Frames TF para conversão câmara → mundo
        self.declare_parameter("camera_frame", "sim_camera")
        self.declare_parameter("world_frame",  "base")

        # Pose da câmara no mundo (usada para publicar TF estático se não existir)
        # Posição (metros)
        self.declare_parameter("cam_pos_x",  0.0)
        self.declare_parameter("cam_pos_y", -0.2)
        self.declare_parameter("cam_pos_z",  0.8)
        # Orientação em quaternion — 180° em torno de Y: câmara olha para baixo
        # com eixo Y alinhado com o mundo (cam_X=-world_X, cam_Y=world_Y, cam_Z=-world_Z)
        self.declare_parameter("cam_quat_x", 0.0)
        self.declare_parameter("cam_quat_y", 1.0)
        self.declare_parameter("cam_quat_z", 0.0)
        self.declare_parameter("cam_quat_w", 0.0)

        image_topic      = self.get_parameter("image_topic").value
        apple_pose_topic = self.get_parameter("apple_pose_topic").value
        yolo_model_name  = self.get_parameter("yolo_model").value

        # ── YOLO ─────────────────────────────────────────────────────────────
        self.get_logger().info(f"A carregar modelo YOLO: {yolo_model_name} ...")
        self.model = YOLO(yolo_model_name)
        self.get_logger().info("Modelo YOLO carregado.")

        # ── ROS publishers / subscribers ──────────────────────────────────
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        self.pose_pub = self.create_publisher(Pose, apple_pose_topic, 10)

        self.bridge = CvBridge()

        # ── TF2 ───────────────────────────────────────────────────────────
        self.tf_buffer    = tf2_ros.Buffer()
        self.tf_listener  = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_static_bc = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        # ── Estado do rato ────────────────────────────────────────────────
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_clicked = False

        cv2.namedWindow("Apple Detector - YOLO")
        cv2.setMouseCallback("Apple Detector - YOLO", self._mouse_handler)

        self.get_logger().info(
            f"AppleDetector ativo | image={image_topic} | pose={apple_pose_topic}"
        )

    # ── TF estático da câmara ─────────────────────────────────────────────
    def _publish_camera_tf(self):
        """Publica uma TF estática world → camera_frame com base nos parâmetros."""
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter("world_frame").value
        t.child_frame_id  = self.get_parameter("camera_frame").value
        t.transform.translation.x = float(self.get_parameter("cam_pos_x").value)
        t.transform.translation.y = float(self.get_parameter("cam_pos_y").value)
        t.transform.translation.z = float(self.get_parameter("cam_pos_z").value)
        t.transform.rotation.x    = float(self.get_parameter("cam_quat_x").value)
        t.transform.rotation.y    = float(self.get_parameter("cam_quat_y").value)
        t.transform.rotation.z    = float(self.get_parameter("cam_quat_z").value)
        t.transform.rotation.w    = float(self.get_parameter("cam_quat_w").value)
        self.tf_static_bc.sendTransform(t)
        self.get_logger().info(
            f"TF estático publicado: {t.header.frame_id} → {t.child_frame_id} "
            f"pos=({t.transform.translation.x:.3f}, "
            f"{t.transform.translation.y:.3f}, "
            f"{t.transform.translation.z:.3f})"
        )

    # ── Rato ──────────────────────────────────────────────────────────────
    def _mouse_handler(self, event, x, y, flags, param):
        """Regista posição e clique esquerdo do rato."""
        self.mouse_x = x
        self.mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True

    # ── Utilitários ───────────────────────────────────────────────────────
    @staticmethod
    def _quat_identity():
        """Quaternion identidade (sem rotação) — adequado para objetos redondos."""
        return (0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def _point_in_box(px, py, x1, y1, x2, y2) -> bool:
        return x1 <= px <= x2 and y1 <= py <= y2

    @staticmethod
    def _box_center(x1, y1, x2, y2):
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    # ── Callback principal ───────────────────────────────────────────────
    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        conf_thr   = float(self.get_parameter("conf_threshold").value)
        z_dist     = float(self.get_parameter("z_dist").value)
        fx         = float(self.get_parameter("fx").value)
        fy         = float(self.get_parameter("fy").value)
        cx         = float(self.get_parameter("cx").value)
        cy         = float(self.get_parameter("cy").value)
        show_conf  = bool(self.get_parameter("show_conf").value)
        debug_mode = bool(self.get_parameter("debug_mode").value)
        use_hsv    = bool(self.get_parameter("use_hsv").value)

        display = cv_image.copy()

        # ── 1. YOLO ──────────────────────────────────────────────────────
        results = self.model(cv_image, verbose=False)[0]
        class_names = self.model.names  # dict {id: name}

        apple_found = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            if debug_mode:
                # Modo diagnóstico: desenha TUDO acima de 10% com classe
                if conf < 0.10:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (180, 180, 180)  # cinzento para outras classes
                if cls_id == COCO_APPLE_CLASS:
                    color = (0, 200, 50)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                lbl = f"{class_names.get(cls_id, cls_id)} {conf*100:.0f}%"
                cv2.putText(display, lbl, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if cls_id == COCO_APPLE_CLASS:
                    self.get_logger().info(
                        f"[DEBUG] APPLE detetada: conf={conf*100:.0f}% | "
                        f"bbox=({x1},{y1},{x2},{y2})"
                    )
                continue

            # Modo normal: filtrar só maçãs
            if cls_id != COCO_APPLE_CLASS or conf < conf_thr:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            apple_found = True
            self._draw_and_publish(display, x1, y1, x2, y2,
                                   conf, show_conf, z_dist, fx, fy, cx, cy,
                                   source="YOLO")

        # ── 2. Fallback HSV (maçãs sintéticas vermelhas/verdes) ──────────
        if use_hsv and not apple_found and not debug_mode:
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Vermelho (dois intervalos de hue)
            m1 = cv2.inRange(hsv, np.array([0,  80, 50]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([160, 80, 50]), np.array([180, 255, 255]))
            # Verde maçã
            m3 = cv2.inRange(hsv, np.array([35, 60, 50]), np.array([85, 255, 255]))
            mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
            mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:  # ignorar ruído pequeno
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # Verificar circularidade (maçã ≈ redonda)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * math.pi * area / (perimeter ** 2)
                if circularity < 0.4:  # muito não-circular → ignorar
                    continue

                apple_found = True
                x1, y1, x2, y2 = x, y, x + w, y + h
                self._draw_and_publish(display, x1, y1, x2, y2,
                                       conf=1.0, show_conf=False,
                                       z_dist=z_dist, fx=fx, fy=fy,
                                       cx=cx, cy=cy, source="HSV")

        if not apple_found and not debug_mode and self.mouse_clicked:
            self.mouse_clicked = False

        # Legenda de modo no canto
        mode_txt = "[DEBUG]" if debug_mode else ("[HSV+YOLO]" if use_hsv else "[YOLO]")
        cv2.putText(display, mode_txt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        cv2.imshow("Apple Detector - YOLO", display)
        cv2.waitKey(1)

    def _draw_and_publish(self, display, x1, y1, x2, y2,
                          conf, show_conf, z_dist, fx, fy, cx, cy, source):
        """Desenha a bounding box e publica Pose no clique (em coordenadas do mundo)."""
        u, v = self._box_center(x1, y1, x2, y2)
        cursor_inside = self._point_in_box(
            self.mouse_x, self.mouse_y, x1, y1, x2, y2
        )
        box_color = (0, 255, 255) if cursor_inside else (0, 200, 50)

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)
        cv2.circle(display, (u, v), 6, (255, 60, 0), -1)

        label = f"Maca [{source}] {conf*100:.0f}%" if show_conf else f"Maca [{source}]"
        cv2.putText(display, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

        # ── Posição em coordenadas da câmara ─────────────────────────────
        x_cam = (u - cx) * z_dist / fx
        y_cam = (v - cy) * z_dist / fy

        self.get_logger().debug(
            f"[cam_frame] x={x_cam:.3f} y={y_cam:.3f} z={z_dist:.3f}  "
            f"pixel=({u},{v}) centre=({int(cx)},{int(cy)})"
        )

        # ── Converter para referencial do mundo via TF2 ───────────────────
        camera_frame = self.get_parameter("camera_frame").value
        world_frame  = self.get_parameter("world_frame").value

        pt_cam = PointStamped()
        pt_cam.header.frame_id = camera_frame
        pt_cam.header.stamp    = self.get_clock().now().to_msg()
        pt_cam.point.x = float(x_cam)
        pt_cam.point.y = float(y_cam)
        pt_cam.point.z = float(z_dist)

        try:
            pt_world = self.tf_buffer.transform(
                pt_cam, world_frame, timeout=Duration(seconds=0.1)
            )
            wx, wy, wz = pt_world.point.x, pt_world.point.y, pt_world.point.z
            active_frame = world_frame
        except Exception as e:
            self.get_logger().warn(
                f"TF falhou ({e}) — publicando em coordenadas da câmara"
            )
            wx, wy, wz = x_cam, y_cam, z_dist
            active_frame = camera_frame

        # Mostrar coordenadas no mundo na imagem
        coord_txt = f"{active_frame}: X={wx:.2f} Y={wy:.2f} Z={wz:.2f}"
        cv2.putText(display, coord_txt, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        if self.mouse_clicked and cursor_inside:
            qx, qy, qz, qw = self._quat_identity()
            pose_msg = Pose()
            pose_msg.position.x = wx
            pose_msg.position.y = wy
            pose_msg.position.z = wz
            pose_msg.orientation.x = qx
            pose_msg.orientation.y = qy
            pose_msg.orientation.z = qz
            pose_msg.orientation.w = qw
            self.pose_pub.publish(pose_msg)
            self.get_logger().info(
                f"[APPLE/{source}] Publicado em [{active_frame}]: "
                f"X={wx:.3f} m  Y={wy:.3f} m  Z={wz:.3f} m  conf={conf*100:.0f}%"
            )
            self.mouse_clicked = False


def main(args=None):
    rclpy.init(args=args)
    node = AppleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
