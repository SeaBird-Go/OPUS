'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-05-27 21:55:54
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pickle
from nuscenes import NuScenes


def add_liarseg_info(split=["train"]):
    version = 'v1.0-trainval'
    data_root = 'data/nuscenes'
    nuscenes = NuScenes(version, data_root)

    assert isinstance(split, list), "split should be a list of strings, e.g., ['train', 'val']"

    for _split in split:
        # load the dataset
        pkl_file = f"data/nuscenes/nuscenes_infos_{_split}_sweep.pkl"
        print(f"Loading dataset for {_split} split in {pkl_file}...")
        with open(pkl_file, "rb") as fp:
            dataset = pickle.load(fp)

        # add lidarseg info
        infos = []
        for info in tqdm(dataset['infos']):
            token = info['token']
            lidar_token = nuscenes.get('sample', token)['data']['LIDAR_TOP']

            lidarseg_label = os.path.join(nuscenes.dataroot, 
                                        nuscenes.get('lidarseg', lidar_token)['filename'])
            info['lidarseg'] = lidarseg_label
            infos.append(info)

        dataset['infos'] = infos

        # save the dataset with lidarseg info
        save_path = f"data/nuscenes/nuscenes_infos_{_split}_sweep_lidarseg.pkl"
        with open(save_path, "wb") as fp:
            pickle.dump(dataset, fp)


if __name__ == '__main__':
    add_liarseg_info(split=["train", "val"])
