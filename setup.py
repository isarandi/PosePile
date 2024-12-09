from setuptools import setup
import os
import os.path as osp


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
        'addict',
        'transforms3d',
        'numpy',
        'more-itertools',
        'msgpack-numpy',
        'pycocotools',
        'multiprocessing-utils',
#        'cameralib @ git+https://github.com/isarandi/cameralib.git',
#        'boxlib @ git+https://github.com/isarandi/boxlib.git',
#        'rlemasklib @ git+https://github.com/isarandi/rlemasklib.git',
#        'simplepyutils @ git+https://github.com/isarandi/simplepyutils.git',
    ]
)
