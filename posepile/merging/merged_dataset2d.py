import argparse

import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets2d as ds2d
from posepile import joint_filtering
from posepile.joint_info import JointInfo
from posepile.merging.merged_dataset3d import merge_joint_infos_of_datasets
from posepile.util import TEST, TRAIN, VALID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    spu.initialize(parser)

    if FLAGS.name == 'huge2d':
        make_huge2d()
    else:
        raise ValueError()


@spu.picklecache('huge2d.pkl', min_time="2022-08-06T14:30:53")
def make_huge2d(separate=True, no_torso=False, no_face=False):
    import posepile.ds.mpii.main
    mpii = posepile.ds.mpii.main.make_mpii_yolo(filter_joints=False)

    coco = ds2d.get_dataset('coco', single_person=False)
    posetrack = ds2d.get_dataset('posetrack')
    jrdb2d = ds2d.get_dataset('jrdb')

    print('Merging...')
    ds = merge_datasets([
        [mpii, [(0, 0)], 'mpii' if separate else ''],
        [coco, [(0, 0), (1, 0)], 'coco' if separate else ''],
        [posetrack, [(0, 0)], 'posetrack' if separate else ''],
        [jrdb2d, [(0, 0)], 'jrdb' if separate else ''],
    ])
    print('Training set size:', len(ds.examples[0]))

    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'.split(',')
    body_joint_ids = [i for name, i in ds.joint_info.ids.items()
                      if any(name.startswith(x) for x in body_joint_names)]

    def n_valid_body_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    ds.examples[TRAIN] = [ex for ex in ds.examples[TRAIN] if n_valid_body_joints(ex) > 6]

    if not separate:
        if no_torso:
            joint_info_used = JointInfo(
                'rank,rkne,lkne,lank,rwri,relb,lelb,lwri,leye,reye,lear,rear,nose',
                'relb-rwri,rkne-rank,rear-reye-nose')
        elif no_face:
            joint_info_used = JointInfo(
                'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri',
                'rsho-relb-rwri,rhip-rkne-rank')
        else:
            joint_info_used = JointInfo(
                'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri,leye,reye,lear,rear,'
                'nose',
                'rsho-relb-rwri,rhip-rkne-rank,rear-reye-nose')
        return joint_filtering.convert_dataset(ds, joint_info_used)

    return ds


def merge_datasets(datasets_with_uses_suf):
    merged_joint_info = merge_joint_infos_of_datasets(datasets_with_uses_suf)

    for i_ds in range(len(datasets_with_uses_suf)):
        datasets_with_uses_suf[i_ds][0] = joint_filtering.convert_dataset(
            datasets_with_uses_suf[i_ds][0], merged_joint_info, update_bones=False)

    examples = {TRAIN: [], VALID: [], TEST: []}
    for ds, uses, suf in datasets_with_uses_suf:
        for take, use_as in uses:
            examples[use_as] += ds.examples[take]

    return ds2d.Pose2DDataset(merged_joint_info, examples[0], examples[1], examples[2])


if __name__ == '__main__':
    main()
