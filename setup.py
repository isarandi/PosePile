from setuptools import setup
import os
import os.path as osp

try:
    dependencies_managed_by_conda = os.environ['DEPENDENCIES_MANAGED_BY_CONDA'] == '1'
except KeyError:
    dependencies_managed_by_conda = False

setup(
    name='posepile',
    version='0.1.0',
    author='István Sárándi',
    author_email='sarandi@vision.rwth-aachen.de',
    packages=['posepile'],
    scripts=[],
    license='LICENSE',
    description='',
    python_requires='>=3.6',
    install_requires=[] if dependencies_managed_by_conda else [
        'tensorflow',
        'attrdict',
        'transforms3d',
        'numpy',
        'more-itertools',
        'cameralib @ git+https://github.com/isarandi/cameralib.git',
        'boxlib @ git+https://github.com/isarandi/boxlib.git',
        'rlemasklib @ git+https://github.com/isarandi/rlemasklib.git',
        'simplepyutils @ git+https://github.com/isarandi/simplepyutils.git',
        'throttledpool @ git+https://github.com/isarandi/throttledpool.git',
    ]
)
