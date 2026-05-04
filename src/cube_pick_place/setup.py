from setuptools import find_packages, setup


package_name = 'cube_pick_place'


setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Apple detection with YOLOv8 and HSV fallback for SO-ARM101.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'apple_detector = cube_pick_place.apple_detector:main',
        ],
    },
)
