import os
import os.path as osp
import os.path

import numpy as np
from posepile.paths import DATA_ROOT


def random_uniform_disc(rng):
    """Samples a random 2D point from the unit disc with a uniform distribution."""
    angle = rng.uniform(-np.pi, np.pi)
    radius = np.sqrt(rng.uniform(0, 1))
    return radius * np.array([np.cos(angle), np.sin(angle)])


def ensure_absolute_path(path, root=DATA_ROOT):
    if not root:
        return path

    if osp.isabs(path):
        return path
    else:
        return osp.join(root, path)

