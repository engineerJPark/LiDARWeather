import os
from os import path as osp
from pathlib import Path

import mmengine

cor_list = [
    'val',
    'snow',
    'light_fog',
    'dense_fog',
    'rain'
]
ROOTDIR = '/data/SemanticSTF'
VALTEXTDIR = ROOTDIR + '/val/val.txt'


with open(VALTEXTDIR, 'r') as f:
    val_list = f.readlines()
    val_list = [x.strip() for x in val_list]
    sample_idx, sample_cor = [], []
    for i in range(len(val_list)):
        sample_idx.append(val_list[i].split(',')[0])
        sample_cor.append(val_list[i].split(',')[1])

def get_semanticstf_info(cor, sample_idx, sample_cor):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'SemanticSTF'},
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

    sample_idx, sample_cor: list of sample index and correspoding corruption
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='SemanticSTF')
    data_list = []
    for j in range(0, len(sample_idx)):
        if cor != 'val' and sample_cor[j] != cor:
            continue
        data_list.append({
            'lidar_points': {
                'lidar_path':
                osp.join('velodyne', sample_idx[j] + '.bin'),
                'num_pts_feats': 4
            },
            'pts_semantic_mask_path':
            osp.join('labels', sample_idx[j] + '.label'),
            'sample_id': str(0) + str(j)
        })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_semantickitti_info_file(pkl_prefix, save_path):
    """Create info file of SemanticSTF dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate info of SemanticSTF.')
    save_path = Path(save_path)
    
    for cor in cor_list:
        semanticstf_infos_train = get_semanticstf_info(cor, sample_idx, sample_cor)
        if cor == 'val':
            filename = save_path / f'{pkl_prefix}_infos_val.pkl'
        else:
            filename = save_path / f'{pkl_prefix}_infos_val_{cor}.pkl'
        import pdb; pdb.set_trace()
        print(f'SemanticSTF info train file is saved to {filename}')
        mmengine.dump(semanticstf_infos_train, filename)