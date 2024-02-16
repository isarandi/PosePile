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
    packages=['posepile','posepile.util'],
    scripts=[],
    license='LICENSE',
    description='',
    python_requires='>=3.6',
    install_requires=[] if dependencies_managed_by_conda else [
        'tensorflow',
        'attrdict',
        'transforms3d',
        'numpy',
        'scipy',
        'more-itertools',
        'msgpack-numpy',
        'numba',
        'jpeg4py',
        'imageio[ffmpeg]',
        'cameralib @ git+https://github.com/isarandi/cameralib.git',
        'boxlib @ git+https://github.com/isarandi/boxlib.git',
        'rlemasklib @ git+https://github.com/isarandi/rlemasklib.git',
        'simplepyutils @ git+https://github.com/isarandi/simplepyutils.git',
        'humcentr-cli @ git+https://github.com/isarandi/humcentr-cli.git',
        'barecat @ git+https://github.com/isarandi/BareCat.git',
        'posepile.util @ git+https://github.com/isarandi/PosePile.git'
    ]
)
