# Deformable 3D Point Cloud Perturbations using Cage-based Deformation for Semantic Consistency

**Kyo Seok Lee and Hak Gu Kim**  
**IEEE Transactions on Information Forensics and Security, 2026**

[Paper](https://ieeexplore.ieee.org/document/11534487) | [Project Page](https://irislab.cau.ac.kr/Deformable-3D-ADV/)

Research code for semantically consistent adversarial point-cloud deformation. The workflow is **shape-deformation training → adversarial decoder training → evaluation**.

## Installation

Use Linux, an NVIDIA GPU and a compatible CUDA development toolkit (`nvcc`). The original container used PyTorch 1.10.0 / CUDA 11.3; the legacy PyMesh wheel targets Python 3.7. The recipe below has not been validated in this cleanup.

```bash
git clone https://github.com/kslee990614/Deformable-3-D-Point-Cloud-Perturbations-Using-Cage-Based-Deformation-for-Semantic-Consistency.git Deformable-3D-ADV
cd Deformable-3D-ADV
conda create -n deformable3d python=3.7 -y
conda activate deformable3d
conda install pytorch=1.10.0 torchvision=0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install faiss-gpu -c pytorch -y
python -m pip install "pip<24.1" "setuptools<68" wheel ninja
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e ./pytorch_points
python tools/check_environment.py
```

The conda CUDA runtime does not provide `nvcc`. If PyMesh installation fails, follow its [source installation guide](https://pymesh.readthedocs.io/en/latest/installation.html).

## Data preparation

Obtain [ShapeNet-Part](https://www.shapenet.org/), [ModelNet40](https://modelnet.cs.princeton.edu/download.html), or [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/). Use the original train/test splits and classifier label order.

`prepare_data.py` creates the category folders, XYZ TXT files, train/test JSONs, class mapping and preprocessing record. It accepts existing TXT folders with split JSONs, combined/separate NPZ splits, or HDF5 files with `data`/`label` keys. It never overwrites an existing destination or invents a new split.

```bash
# Existing ShapeNet-Part point folders and official split JSONs
python prepare_data.py --dataset SHAPENET --input_dir raw/shapenetpart --output data/shapenet --num_point 1024 --sampling fps

# NPZ keys: train_pc, train_label, test_pc, test_label
python prepare_data.py --dataset MN40 --input_npz raw/modelnet40.npz --output data/mn40 --num_point 1024 --sampling fps

# ScanObjectNN object-only HDF5; names must follow the victim's label order
python prepare_data.py --dataset SCANOBJECTNN --train_h5 raw/training_objectdataset.h5 --test_h5 raw/test_objectdataset.h5 --class_names raw/scanobjectnn_classes.json --output data/scanobjectnn --num_point 1024 --sampling fps
```

For separate NPZ files, use `--train_npz train.npz --test_npz test.npz` instead of `--input_npz`; each file uses its corresponding `train_*` or `test_*` keys. Trusted historical object-array NPZ additionally requires `--allow_pickle_data`. Raw OFF/OBJ meshes must first be sampled into point clouds; this converter does not sample mesh surfaces.

Output layout:

```text
data/<dataset>/
  <category>/train_000000.txt
  <category>/test_000000.txt
  train_test_split/shuffled_train_file_list.json
  train_test_split/shuffled_test_file_list.json
  class_names.json
  synsetoffset2category.txt
  preprocessing.json
```

Each TXT contains 1,024 XYZ points. FPS is explicit; normalization defaults to `none`. Add `--normalization unit_sphere` or `bbox_ground` only when it matches the trained models. The legacy ShapeNet loader applies bounding-box normalization, Y=-1 grounding and stage-one scale augmentation; shared MN40/ScanObjectNN loading preserves prepared coordinates. Existing checkpoints must use compatible preprocessing.

Inspect the result before training:

```bash
python tools/inspect_data.py --data_dir data/scanobjectnn
```

Empty files and overlapping split IDs are rejected. Split-prefixed output names avoid filename collisions; inputs without original sample IDs still rely on the correctness of the supplied official split membership. ScanObjectNN needs a 15-name JSON array in the classifier's numeric label order, not an inferred alphabetical order.

## Checkpoints

Provide the matching victim and category-specific cage weights. These are not bundled and no public checkpoint download was supplied. The release does not include victim-classifier training; exact result reproduction requires the original victim weights.

```text
checkpoints/scanobjectnn/victims/pointnet.pth
checkpoints/scanobjectnn/stage1/bag/net_final.pth
checkpoints/scanobjectnn/stage2/bag/net_final.pth
```

PointNet weights must match `feature_transform=False`. Cage flags must match the checkpoint architecture; the supplied commands use bottleneck 256 and two folds. Loading is strict.

## Training

The following direct commands demonstrate ScanObjectNN / PointNet / Bag. Stage one is an example configuration; stage two preserves the supplied Bag epoch count and Chamfer + MVC settings.

### Stage 1: shape deformation

```bash
python cage_deformer_3d.py --dataset SCANOBJECTNN --data_dir data/scanobjectnn --class_names data/scanobjectnn/class_names.json --data_cat bag --phase train --batch_size 32 --nepochs 50 --num_point 1024 --bottleneck_size 256 --n_fold 2 --loss CD --name scanobjectnn_bag --loss_weight 1 --mvc_weight 1 --sym_weight 0 --p2f_weight 0.1 --snormal_weight 0.1 --full_net --log_dir log/scanobjectnn/stage1
```

The run saves `net_final.pth` in a timestamped folder below `--log_dir`. Pass that actual path to stage two, or copy the file to the stage-one path shown above.

### Stage 2: adversarial decoder

```bash
python cage_deformer_3d.py --dataset SCANOBJECTNN --data_dir data/scanobjectnn --class_names data/scanobjectnn/class_names.json --data_cat bag --phase adv_train --batch_size 32 --nepochs 347 --num_point 1024 --bottleneck_size 256 --n_fold 2 --loss CD --name scanobjectnn_bag --loss_weight 1 --mvc_weight 1 --sym_weight 0 --p2f_weight 0 --snormal_weight 0 --full_net --ckpt checkpoints/scanobjectnn/stage1/bag/net_final.pth --classifier pointnet --classifier_ckpt checkpoints/scanobjectnn/victims/pointnet.pth --log_dir log/scanobjectnn/stage2
```

Only `nd_decoder` is trained. The encoder, cage refinement, merger and victim are frozen. `--lr` defaults to 0.001. This unified entry replaces the missing experiment-specific `_pn.py` with `--classifier pointnet`.

## Evaluation

Use the completed **stage-two** checkpoint, copying it to the illustrated path or passing its actual run path:

```bash
python cage_deformer_3d.py --dataset SCANOBJECTNN --data_dir data/scanobjectnn --class_names data/scanobjectnn/class_names.json --data_cat bag --phase adv_test --batch_size 1 --num_point 1024 --bottleneck_size 256 --n_fold 2 --loss CD --full_net --data_max -1 --ckpt checkpoints/scanobjectnn/stage2/bag/net_final.pth --classifier pointnet --classifier_ckpt checkpoints/scanobjectnn/victims/pointnet.pth --log_dir log/scanobjectnn/eval/bag
```

Evaluation saves predictions, ASR metrics and deformed point clouds under the requested log directory. No adversarial optimization runs at test time; geometric cage adjustment still iterates inside the forward call.

For another dataset, change the dataset, root, category and matching checkpoints in these commands:

| Dataset | Flag | Root | Category example | Victim classes |
| --- | --- | --- | --- | --- |
| ShapeNet-Part | `SHAPENET` | `data/shapenet` | `Bag` | 16 |
| ModelNet40 | `MN40` | `data/mn40` | `chair` | 40 |
| ScanObjectNN | `SCANOBJECTNN` | `data/scanobjectnn` | `bag` | 15 |

Keep `--sym_weight 0` for shared MN40/ScanObjectNN loading. Epoch counts and loss settings remain experiment-specific. The MN40/ScanObjectNN adapters are new integrations; their equivalence to the missing original server implementations and GPU results has not been verified. Complete PCT/defense/Uniform/KNN experiment drivers are not included.

## Citation

```bibtex
@article{lee2026deformable,
  title={Deformable 3D Point Cloud Perturbations using Cage-based Deformation for Semantic Consistency},
  author={Lee, Kyo Seok and Kim, Hak Gu},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/11534487}
}
```

License notices are preserved in `LICENSE`, `pytorch_points/license`, and the original source headers.

This repository is based on [yifita/deep_cage](https://github.com/yifita/deep_cage).
