"""Export official train/test point-cloud splits into the shared TXT layout."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from dataset_io import (COUNTS, SHAPENET_FOLDERS, class_names, PointCloudIndex,
                        prepare_points, sample_seed)


def npz_samples(path, split, allow_pickle=False):
    with np.load(path, allow_pickle=allow_pickle) as data:
        points = data[split + '_pc']
        labels = np.asarray(data[split + '_label']).reshape(-1)
        if len(points) != len(labels) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('Mismatched points/labels or noninteger labels')
        ids = data[split + '_id'].astype(str) if split + '_id' in data else None
        if ids is not None and len(ids) != len(labels):
            raise ValueError('Sample ID count mismatch')
        for i, (point, label) in enumerate(zip(points, labels)):
            yield point, int(label), str(ids[i]) if ids is not None else None


def h5_samples(paths):
    import h5py
    for path in paths:
        with h5py.File(path, 'r') as data:
            points, labels = data['data'], data['label']
            if len(points) != len(labels) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError('Mismatched HDF5 points/labels or noninteger labels')
            for i in range(len(points)):
                label = np.asarray(labels[i]).reshape(-1)
                if len(label) != 1:
                    raise ValueError('Expected one class label per point cloud')
                yield points[i], int(label[0]), None


def folder_samples(args, split):
    index = PointCloudIndex(args.input_dir, args.dataset, split, names_path=args.class_names)
    for i in range(len(index.records)):
        yield index.points(i)


def export(args):
    names = class_names(args.dataset, args.class_names)
    folders = SHAPENET_FOLDERS if args.dataset == 'SHAPENET' else names
    if any(Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name for name in folders):
        raise ValueError('Class names must be plain folder names')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists: {}'.format(output))
    # Check original IDs too, before adding split prefixes to exported filenames.
    original_ids = {'train': set(), 'test': set()}
    provenance = {'train': [], 'test': []}
    counts = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=str(output.parent)) as temp:
        prepared = Path(temp) / 'prepared'
        prepared.mkdir()
        manifests = {}
        for split in ('train', 'test'):
            if args.input_dir:
                samples = folder_samples(args, split)
            elif args.input_npz or args.train_npz:
                path = args.input_npz or getattr(args, split + '_npz')
                samples = npz_samples(path, split, args.allow_pickle_data)
            else:
                samples = h5_samples(getattr(args, split + '_h5'))
            ids = []
            for i, (points, label, original_id) in enumerate(samples):
                if not 0 <= label < COUNTS[args.dataset]:
                    raise ValueError('Label outside dataset range: {}'.format(label))
                if original_id is not None:
                    if original_id in original_ids[split]:
                        raise ValueError('Duplicate source ID in {}: {}'.format(split, original_id))
                    original_ids[split].add(original_id)
                identifier = '{}/{}_{:06d}'.format(folders[label], split, i)
                xyz = prepare_points(points, args.num_point, args.sampling,
                                     args.normalization, sample_seed(args.seed, identifier))
                target = prepared / (identifier + '.txt')
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(target, xyz, fmt='%.9g')
                ids.append('shape_data/' + identifier)
                provenance[split].append({'output': identifier, 'label': label, 'source_id': original_id})
            if not ids:
                raise ValueError('Empty {} split'.format(split))
            manifests[split] = ids
            counts[split] = len(ids)
        if original_ids['train'] & original_ids['test']:
            raise ValueError('Train/test source IDs overlap; restore official split IDs')
        splits = prepared / 'train_test_split'
        splits.mkdir()
        for split, ids in manifests.items():
            (splits / ('shuffled_' + split + '_file_list.json')).write_text(json.dumps(ids, indent=2), encoding='utf-8')
        (prepared / 'class_names.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
        (prepared / 'synsetoffset2category.txt').write_text(''.join('{} {}\n'.format(n,f) for n,f in zip(names,folders)), encoding='utf-8')
        metadata = {'arguments': vars(args), 'counts': counts, 'class_names': names,
                    'samples': provenance, 'split_policy': 'preserve supplied train/test membership'}
        (prepared / 'preprocessing.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        prepared.rename(output)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=str.upper, choices=list(COUNTS), required=True)
    p.add_argument('--input_npz', help='One NPZ with both train_pc/train_label and test_pc/test_label')
    p.add_argument('--train_npz')
    p.add_argument('--test_npz')
    p.add_argument('--input_dir', help='Existing XYZ folders with official split JSONs')
    p.add_argument('--train_h5', nargs='+', help='Official training HDF5 files with data/label keys')
    p.add_argument('--test_h5', nargs='+', help='Official test HDF5 files with data/label keys')
    p.add_argument('--class_names', help='Class-name JSON list in classifier numeric label order; required for ScanObjectNN')
    p.add_argument('--output', required=True, help='New destination folder; never overwritten')
    p.add_argument('--num_point', type=int, default=1024)
    p.add_argument('--sampling', choices=['none', 'fps', 'random'], default='none')
    p.add_argument('--normalization', choices=['none', 'unit_sphere', 'bbox_ground'], default='none')
    p.add_argument('--seed', type=int, default=24)
    p.add_argument('--allow_pickle_data', action='store_true', help='Permit trusted historical object-array NPZ')
    args = p.parse_args()
    sources = [bool(args.input_npz), bool(args.input_dir), bool(args.train_npz or args.test_npz), bool(args.train_h5 or args.test_h5)]
    if sum(sources) != 1:
        p.error('Choose one source: --input_npz, --input_dir, train/test NPZ pair, or train/test HDF5 pair')
    if bool(args.train_npz) != bool(args.test_npz) or bool(args.train_h5) != bool(args.test_h5):
        p.error('Provide both train and test inputs')
    print(json.dumps(export(args)))


if __name__ == '__main__':
    main()
