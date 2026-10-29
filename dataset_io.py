"""Shared NumPy data I/O for ShapeNet-Part, ModelNet40 and ScanObjectNN.

No CUDA imports. Existing prepared coordinates are preserved unless a transform
is explicitly selected. Class ordering is part of the classifier checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SHAPENET_CLASSES = ['Airplane','Bag','Cap','Car','Chair','Earphone','Guitar','Knife',
                   'Lamp','Laptop','Motorbike','Mug','Pistol','Rocket','Skateboard','Table']
SHAPENET_FOLDERS = ['02691156','02773838','02954340','02958343','03001627','03261776',
                   '03467517','03624134','03636649','03642806','03790512','03797390',
                   '03948459','04099429','04225987','04379243']
MODELNET_CLASSES = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car',
                   'chair','cone','cup','curtain','desk','door','dresser','flower_pot',
                   'glass_box','guitar','keyboard','lamp','laptop','mantel','monitor',
                   'night_stand','person','piano','plant','radio','range_hood','sink',
                   'sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
COUNTS = {'SHAPENET':16, 'MN40':40, 'SCANOBJECTNN':15}


def class_names(dataset, path=None):
    if path:
        names = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(names,list) or not all(isinstance(x,str) for x in names):
            raise ValueError('class_names must be a JSON list in classifier label order')
    elif dataset == 'SHAPENET':
        names = SHAPENET_CLASSES
    elif dataset == 'MN40':
        names = MODELNET_CLASSES
    else:
        raise ValueError('ScanObjectNN folder input requires --class_names: exact classifier label order')
    if len(names) != COUNTS[dataset] or len(set(x.casefold() for x in names)) != len(names):
        raise ValueError('Wrong class count or duplicate class names')
    return names


def split_path(root, split):
    root = Path(root)
    for parent in (root/'train_test_split', root):
        p = parent/('shuffled_{}_file_list.json'.format(split))
        if p.is_file() and p.stat().st_size:
            return p
    raise FileNotFoundError('Nonempty {} split JSON missing under {}'.format(split,root))


def split_ids(root, split):
    values = json.loads(split_path(root,split).read_text(encoding='utf-8'))
    if not isinstance(values,list) or not values:
        raise ValueError('Split JSON must be a nonempty list')
    ids = []
    for value in values:
        parts = str(value).replace('\\','/').split('/')
        if len(parts)<2 or any(p in ('','..','.') for p in parts[-2:]):
            raise ValueError('Invalid split identifier: {}'.format(value))
        ids.append('/'.join(parts[-2:]).removesuffix('.txt') if hasattr(str,'removesuffix')
                   else '/'.join(parts[-2:])[:-4] if value.endswith('.txt') else '/'.join(parts[-2:]))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs in {} split'.format(split))
    return ids


def validate_disjoint(root):
    train, test = set(split_ids(root,'train')), set(split_ids(root,'test'))
    overlap = sorted(train & test)
    if overlap:
        raise ValueError('Train/test leakage: {} shared IDs; examples: {}'.format(len(overlap),overlap[:5]))


def read_points(path):
    p = Path(path)
    if p.stat().st_size == 0:
        raise ValueError('Empty point file: {}'.format(path))
    try:
        x = np.loadtxt(p,dtype=np.float32)
    except ValueError:
        x = np.loadtxt(p,dtype=np.float32,delimiter=',')
    if x.ndim != 2 or x.shape[1]<3 or not len(x) or not np.isfinite(x[:,:3]).all():
        raise ValueError('Expected finite N x 3(+) coordinates: {}'.format(path))
    return x[:,:3]


def prepare_points(points, num_point=1024, sampling='none', normalization='none', seed=24):
    if num_point <= 0:
        raise ValueError('num_point must be positive')
    x = np.asarray(points,dtype=np.float32)
    if x.ndim != 2 or x.shape[1]<3 or len(x)==0 or not np.isfinite(x[:,:3]).all():
        raise ValueError('Expected nonempty finite N x 3(+) coordinates')
    x = x[:,:3].copy()
    # Explicitly normalize before sampling, as in the legacy deformer loader.
    if normalization == 'unit_sphere':
        x -= x.mean(0,keepdims=True)
        scale = np.linalg.norm(x,axis=1).max()
    elif normalization == 'bbox_ground':
        x -= (x.max(0,keepdims=True)+x.min(0,keepdims=True))/2
        scale = np.abs(x).max()
    elif normalization == 'none':
        scale = 1.0
    else:
        raise ValueError('Unknown normalization')
    if scale <= 1e-12:
        raise ValueError('Degenerate point cloud')
    x /= scale
    if normalization == 'bbox_ground':
        x[:,1] -= x[:,1].min()+1
    if sampling == 'none':
        if len(x) != num_point:
            raise ValueError('Expected {} points, got {}; select an explicit sampling method'.format(num_point,len(x)))
    elif sampling in ('fps','random'):
        if len(x)<num_point:
            raise ValueError('Too few points; this loader does not silently duplicate points')
        rng = np.random.RandomState(seed)
        if sampling == 'random':
            idx = rng.choice(len(x),num_point,replace=False)
        else:
            idx = np.empty(num_point,dtype=np.int64)
            distances = np.full(len(x),np.inf)
            farthest = rng.randint(len(x))
            for i in range(num_point):
                idx[i]=farthest
                distances=np.minimum(distances,((x-x[farthest])**2).sum(1))
                distances[idx[:i+1]] = -1
                farthest=int(distances.argmax())
        x=x[idx]
    else:
        raise ValueError('Unknown sampling')
    return np.ascontiguousarray(x,dtype=np.float32)


def sample_seed(seed, identifier):
    return (seed+int(hashlib.sha256(identifier.encode()).hexdigest()[:8],16)) % (2**32)


class PointCloudIndex:
    """A dataset-independent list of points, class labels and stable IDs."""
    def __init__(self, root, dataset, split, category='*', names_path=None,
                 allow_pickle=False):
        self.root=Path(root)
        self.dataset=dataset
        self.names = class_names(dataset,names_path) if names_path or dataset!='SCANOBJECTNN' or self.root.is_dir() else None
        self.records=[]
        self.arrays=None
        self.split=split
        if self.root.is_file():
            with np.load(self.root,allow_pickle=allow_pickle) as data:
                self.arrays=data[split+'_pc']
                raw_labels=data[split+'_label']
                labels=np.asarray(raw_labels).reshape(-1)
                if not np.issubdtype(labels.dtype,np.integer):
                    raise ValueError('Class labels must have integer dtype')
                if len(self.arrays)!=len(labels):
                    raise ValueError('Point/label count mismatch')
                if 'train_id' in data and 'test_id' in data:
                    overlap=set(data['train_id'].astype(str).tolist()) & set(data['test_id'].astype(str).tolist())
                    if overlap:
                        raise ValueError('Train/test leakage in NPZ IDs')
                ids=data[split+'_id'].astype(str).tolist() if split+'_id' in data else [split+'_'+str(i) for i in range(len(labels))]
                if len(ids)!=len(labels) or len(set(ids))!=len(ids):
                    raise ValueError('Invalid or duplicate NPZ sample IDs')
                for i,(label,identifier) in enumerate(zip(labels,ids)):
                    self.records.append((identifier,int(label),i))
        else:
            validate_disjoint(self.root)
            folders=SHAPENET_FOLDERS if dataset=='SHAPENET' else self.names
            mapping={folder.casefold():i for i,folder in enumerate(folders)}
            for identifier in split_ids(self.root,split):
                folder=identifier.split('/')[0]
                if folder.casefold() not in mapping:
                    raise ValueError('Unmapped category folder: '+folder)
                path=self.root/(identifier+'.txt')
                if not path.is_file():
                    raise FileNotFoundError(path)
                self.records.append((identifier,mapping[folder.casefold()],path))
        if any(label<0 or label>=COUNTS[dataset] for _,label,_ in self.records):
            raise ValueError('Labels outside dataset class range')
        if category!='*':
            if category.isdigit():
                wanted=int(category)
            else:
                if self.names is None:
                    raise ValueError('Named category needs --class_names')
                wanted=[x.casefold() for x in self.names].index(category.casefold())
            self.records=[r for r in self.records if r[1]==wanted]
        if not self.records:
            raise ValueError('Empty dataset/category')

    def points(self,index):
        identifier,label,key=self.records[index]
        x=self.arrays[key] if self.arrays is not None else read_points(key)
        return x,label,identifier


def main():
    p=argparse.ArgumentParser(description='Inspect/prepare shared point-cloud inputs without CUDA')
    p.add_argument('--dataset',type=str.upper,choices=list(COUNTS),required=True)
    p.add_argument('--data_dir',required=True)
    p.add_argument('--class_names')
    p.add_argument('--split',choices=['train','test'],default='train')
    p.add_argument('--output',help='Optional prepared NPZ for this split; never overwrites')
    p.add_argument('--sampling',choices=['none','random','fps'],default='none')
    p.add_argument('--normalization',choices=['none','bbox_ground','unit_sphere'],default='none')
    p.add_argument('--num_point',type=int,default=1024)
    p.add_argument('--seed',type=int,default=24)
    p.add_argument('--allow_pickle_data',action='store_true',help='Read trusted historical object-array NPZ')
    args=p.parse_args()
    if args.output and Path(args.output).exists():
        p.error('Output already exists')
    index=PointCloudIndex(args.data_dir,args.dataset,args.split,names_path=args.class_names,allow_pickle=args.allow_pickle_data)
    shapes=[];labels=[];ids=[]
    for i in range(len(index.records)):
        x,label,identifier=index.points(i)
        x=prepare_points(x,args.num_point,args.sampling,args.normalization,sample_seed(args.seed,identifier))
        if args.output:
            shapes.append(x);labels.append(label);ids.append(identifier)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,**{args.split+'_pc':np.stack(shapes),args.split+'_label':np.array(labels,dtype=np.int64),args.split+'_id':np.array(ids)})
        metadata={'arguments':vars(args),'class_names_in_label_order':index.names,
                  'samples':len(index.records),'input_coordinates_modified':args.normalization!='none' or args.sampling!='none'}
        Path(str(args.output)+'.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'split':args.split,'samples':len(index.records),'normalization':args.normalization,'sampling':args.sampling}))


if __name__=='__main__':
    main()
