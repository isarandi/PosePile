import os.path as osp

import boxlib
import numpy as np
import pycocotools.coco
import simplepyutils as spu

import posepile.datasets2d as ds2d
import posepile.joint_filtering
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from posepile.util import TEST, TRAIN, VALID


@spu.picklecache('coco.pkl', min_time="2020-02-01T02:53:21")
def make_dataset(single_person=True):
    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear')
    n_joints = joint_info.n_joints
    learning_phase_shortnames = {TRAIN: 'train', VALID: 'val', TEST: 'test'}
    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2
    iou_threshold = 0.1 if single_person else 0.5

    suffix = '' if single_person else '_multi'
    examples_per_phase = {TRAIN: [], VALID: []}
    with spu.ThrottledPool() as pool:
        for example_phase in (TRAIN, VALID):
            phase_shortname = learning_phase_shortnames[example_phase]
            coco_filepath = (
                f'{DATA_ROOT}/coco/annotations/person_keypoints_{phase_shortname}2014.json')
            coco = pycocotools.coco.COCO(coco_filepath)

            impath_to_examples = {}
            for ann in coco.anns.values():
                filename = coco.imgs[ann['image_id']]['file_name']
                image_path = f'{DATA_ROOT}/coco/{phase_shortname}2014/{filename}'

                joints = np.array(ann['keypoints']).reshape([-1, 3])
                visibilities = joints[:, 2]
                coords = joints[:, :2].astype(np.float32).copy()
                n_visible_joints = np.count_nonzero(visibilities == VISIBLE)
                n_occluded_joints = np.count_nonzero(visibilities == OCCLUDED)
                n_labeled_joints = n_occluded_joints + n_visible_joints

                if n_visible_joints >= n_joints / 3 and n_labeled_joints >= n_joints / 2:
                    coords[visibilities == UNLABELED] = np.nan
                    bbox_pt1 = np.array(ann['bbox'][:2], np.float32)
                    bbox_wh = np.array(ann['bbox'][2:4], np.float32)
                    bbox = np.array([*bbox_pt1, *bbox_wh])
                    mask = coco.annToRLE(ann)
                    ex = ds2d.Pose2DExample(image_path, coords, bbox=bbox, mask=mask)
                    impath_to_examples.setdefault(image_path, []).append(ex)

            n_images = len(impath_to_examples)
            for impath, examples in spu.progressbar(impath_to_examples.items(), total=n_images):
                for i_example, example in enumerate(examples):
                    box = boxlib.expand(boxlib.bb_of_points(example.coords), 1.25)
                    if np.max(box[2:]) < 200:
                        continue

                    if single_person:
                        other_boxes = [boxlib.expand(boxlib.bb_of_points(e.coords), 1.25)
                                       for e in examples if e is not example]
                        ious = np.array([boxlib.iou(b, box) for b in other_boxes])
                        usable = np.all(ious < iou_threshold)
                    else:
                        usable = True

                    if usable:
                        new_im_path = impath.replace('coco', 'coco_downscaled' + suffix)
                        without_ext, ext = osp.splitext(new_im_path)
                        new_im_path = f'{without_ext}_{i_example:02d}{ext}'
                        pool.apply_async(
                            make_efficient_example, (example, new_im_path),
                            callback=examples_per_phase[example_phase].append)

    examples_per_phase[TRAIN].sort(key=lambda ex: ex.image_path)
    examples_per_phase[VALID].sort(key=lambda ex: ex.image_path)
    return ds2d.Pose2DDataset(joint_info, examples_per_phase[TRAIN], examples_per_phase[VALID])


@spu.picklecache('coco_reduced.pkl', min_time="2020-02-01T02:53:21")
def make_reduced_dataset(single_person=False, face=True):
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'
    if face:
        joint_names += ',nose,leye,reye,lear,rear'

    edges = 'lelb-lwri,relb-rwri,lhip-lkne-lank,rhip-rkne-rank'
    joint_info = JointInfo(joint_names, edges)
    ds = posepile.joint_filtering.convert_dataset(make_dataset(single_person), joint_info)

    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'.split(',')
    body_joint_ids = [joint_info.ids[name] for name in body_joint_names]

    def n_valid_body_joints(example):
        return np.count_nonzero(
            np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    ds.examples[TRAIN] = [ex for ex in ds.examples[TRAIN] if n_valid_body_joints(ex) > 6]
    return ds
