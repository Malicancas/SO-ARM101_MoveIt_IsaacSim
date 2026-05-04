from setuptools import find_packages, setup


package_name = "so_arm_target_mover"


setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Apple pick & place for SO-ARM101 using MoveIt 2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "apple_pick_place = so_arm_target_mover.apple_pick_place:main",
        ],
    },
)
