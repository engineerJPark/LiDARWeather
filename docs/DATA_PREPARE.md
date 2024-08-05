<img src="../docs/figs/logo.png" align="right" width="20%">

# Data Preparation

### Overall Structure

```
└── data 
    └── sets
        │── semantickitti
        │── SemanticSTF
        │── SemanticKITTI-C
```

<hr>

### SemanticKITTI

To prepare the [SemanticKITTI](http://semantic-kitti.org/index) dataset, download the data, annotations, and other files from http://semantic-kitti.org/dataset. Unpack the compressed file(s) into `/data/sets/semantickitti` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── semantickitti  
    └── sequences
        ├── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    │── 00
        │    │── ···
        │    └── 21
        ├── labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
        │    │── 00
        │    │── ···
        │    └── 10
        ├── calib
        │    │── 00
        │    │── ···
        │    └── 21
        └── semantic-kitti.yaml
```

#### :memo: Create SemanticKITTI Dataset
- For training and evaluation:
  - We support scripts that generate dataset information for training and validation. Create these `.pkl` info files by running:
    ```Shell
    python ./tools/create_data.py semantickitti --root-path ./data/semantickitti --out-dir ./data/semantickitti --extra-tag semantickitti
    ```

<hr>

### SynLiDAR

To prepare the [SynLiDAR](https://github.com/xiaoaoran/SynLiDAR) dataset, follow their instruction to download and re-organize. Your folder structure should end up looking like this:

```
/SynLiDAR/
  └── 00/
    └── velodyne
      └── 000000.bin
      ├── 000001.bin
      ...
    └── labels
      └── 000000.label
      ├── 000001.label
      ...
  ...
  └── annotations.yaml
  └── read_data.py
```

#### :memo: Create SynLiDAR Dataset
- For training and evaluation:
  - We support scripts that generate dataset information for training and validation. Create these `.pkl` info files by running:
    ```Shell
    python ./tools/create_data.py synlidar --root-path ./data/SynLiDAR --out-dir ./data/SynLiDAR --extra-tag synlidar
    ```

<hr>

### SemanticSTF

To prepare the [SemanticSTF](https://github.com/xiaoaoran/SemanticSTF) dataset, follow their instruction to download and re-organize. Your folder structure should end up looking like this:

```
└── SemanticSTF/
    └── train/
        └── velodyne
        └── 000000.bin
        ├── 000001.bin
        ...
        └── labels
        └── 000000.label
        ├── 000001.label
        ...
    └── val/
        ...
    └── test/
        ...
    ...
    └── semanticstf.yaml
```

#### :memo: Create SemanticSTF Dataset
- For training and evaluation:
  - We support scripts that generate dataset information for training and validation. Create these `.pkl` info files by running:
    ```Shell
    python ./tools/create_data.py semanticstf --root-path ./data/SemanticSTF --out-dir ./data/SemanticSTF --extra-tag semanticstf
    ```

<hr>

### SemanticKITTI-C

To prepare the [SemanticKITTI](https://github.com/ldkong1205/Robo3D) dataset, follow their instruction to download and re-organize. Your folder structure should end up looking like this:

```  
└── SemanticKITTI-C  
    ├── fog
    │    ├── light
    │    │     ├── velodyne           
    │    │     └── labels    
    │    ├── moderate
    │    └── heavy
    ├── wet_ground
    ├── snow
    ├── motion_blur
    ├── beam_missing
    ├── crosstalk
    ├── incomplete_echo
    └── cross_sensor
```

#### :memo: Create SemanticKITTI-C Dataset
- For training and evaluation:
  - We support scripts that generate dataset information for training and validation. Create these `.pkl` info files by running:
    ```Shell
    python ./tools/create_data.py semantickitti_c --root-path ./data/SemanticKITTI-C --out-dir ./data/SemanticKITTI-C --extra-tag semantickitti_c
    ```