import functools
import importlib
import itertools

import numpy as np
import simplepyutils as spu

from posepile import joint_filtering
from posepile.joint_info import JointInfo
from posepile.util import TEST, TRAIN, VALID


class Pose2DDataset:
    def __init__(
            self, joint_info, train_examples=None, valid_examples=None, test_examples=None):
        self.joint_info = joint_info
        self.examples = {
            TRAIN: train_examples or [], VALID: valid_examples or [], TEST: test_examples or []}

    def iter_examples(self):
        return itertools.chain(self.examples[TRAIN], self.examples[VALID], self.examples[TEST])


class Pose2DExample:
    def __init__(self, image_path, coords, bbox=None, mask=None, camera=None):
        self.image_path = image_path
        self.coords = coords
        self.bbox = np.asarray(bbox)
        self.mask = mask
        self.camera = camera


@spu.picklecache('many2d.pkl', min_time="2020-02-01T02:53:21")
def make_many2d():
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri,nose,leye,reye,lear,rear'
    edges = 'relb-rwri,rhip-rkne-rank'
    joint_info = JointInfo(joint_names, edges)
    datasets = [
        get_dataset('coco', single_person=False),
        get_dataset('mpii_yolo'),
        get_dataset('posetrack')]
    datasets = [joint_filtering.convert_dataset(ds, joint_info) for ds in datasets]

    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'.split(',')
    body_joint_ids = [joint_info.ids[name] for name in body_joint_names]

    def n_valid_body_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    train_examples = [
        ex for ds in datasets
        for ex in ds.examples[TRAIN]
        if n_valid_body_joints(ex) > 6]
    return Pose2DDataset(joint_info, train_examples)


def make_huge2d():
    import posepile.merging.merged_dataset2d
    return posepile.merging.merged_dataset2d.make_huge2d()


def make_huge2d_common():
    import posepile.merging.merged_dataset2d
    return posepile.merging.merged_dataset2d.make_huge2d(separate=False)


def make_coco_reduced():
    import posepile.ds.coco.main
    return posepile.ds.coco.main.make_reduced_dataset()


def make_mpii_yolo():
    import posepile.ds.mpii.main
    return posepile.ds.mpii.main.make_mpii_yolo()


@functools.lru_cache()
def get_dataset(dataset_name, *args, **kwargs):
    try:
        make_fn = globals()[f'make_{dataset_name}']
    except KeyError:
        try:
            dataset_module = importlib.import_module(f'posepile.ds.{dataset_name}.main')
        except ImportError:
            raise ValueError(f'Dataset {dataset_name} not found.')

        make_fn = dataset_module.make_dataset

    return make_fn(*args, **kwargs)
