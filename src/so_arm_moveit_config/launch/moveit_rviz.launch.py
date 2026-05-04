from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("so101_new_calib", package_name="so_arm_moveit_config")
        .to_moveit_configs()
    )

    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(moveit_config.package_path / "config/moveit.rviz"),
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "rviz_delay",
            default_value="3.0",
            description="Delay RViz startup to avoid stale PlanningScene messages.",
        )
    )

    rviz_parameters = [
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.planning_pipelines,
        moveit_config.robot_description_kinematics,
        moveit_config.joint_limits,
    ]

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=rviz_parameters,
    )

    ld.add_action(
        TimerAction(
            period=LaunchConfiguration("rviz_delay"),
            actions=[rviz_node],
        )
    )

    return ld
