from os import path as osp
from pathlib import Path

import mmengine

total_num = {
    0: 294,
    1: 1426,
    2: 975,
    3: 988,
    4: 1669,
    5: 1492,
    6: 4507,
    7: 1958,
    8: 1147,
    9: 1743,
    10: 1098,
    11: 1426,
    12: 1117,
} ## total 19840
fold_split = {
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'val': [0],
    'test': [0],
}
split_list = ['train', 'valid', 'test']

import os
def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def get_synlidar_info(split):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'SemanticKITTI'},
            'data_list': {
                00000: {
                    'lidar_points':{
                        'lidat_path':'sequences/00/velodyne/000000.bin'
                    },
                    'pts_semantic_mask_path':
                        'sequences/000/labels/000000.labbel',
                    'sample_id': '00'
                },
                ...
            }
        }
        # for j in range(0, total_num[i_folder]):
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='SynLiDAR')
    data_list = []
    for i_folder in fold_split[split]:
        file_list = absoluteFilePaths(osp.join('/data/SynLiDAR', 'sequences', str(i_folder).zfill(2), 'velodyne'))
        for j, file_name in enumerate(file_list):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                        osp.join('sequences',
                                 str(i_folder).zfill(2), 'velodyne',
                                 file_name.split('/')[-1]), # str(j).zfill(6) + '.bin'
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         file_name.split('/')[-1][:-4] + '.label'), ## str(j).zfill(6) + '.label')
                'sample_id':
                str(i_folder) + file_name.split('/')[-1][:-4]
                # str(i_folder) + str(j)
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_synlidar_info_file(pkl_prefix, save_path):
    """Create info file of SemanticKITTI dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate info.')
    save_path = Path(save_path)

    synlidar_infos_train = get_synlidar_info(split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'SynLiDAR info train file is saved to {filename}')
    mmengine.dump(synlidar_infos_train, filename)
    synlidar_infos_val = get_synlidar_info(split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'SynLiDAR info val file is saved to {filename}')
    mmengine.dump(synlidar_infos_val, filename)
    synlidar_infos_test = get_synlidar_info(split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'SynLiDAR info test file is saved to {filename}')
    mmengine.dump(synlidar_infos_test, filename)