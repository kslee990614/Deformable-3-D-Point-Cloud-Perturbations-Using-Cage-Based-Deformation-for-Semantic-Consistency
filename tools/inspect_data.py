"""Read-only folder audit. Does not create splits or repair/drop observations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_io import read_points,split_ids


def audit(root):
    root=Path(root)
    files=sorted(root.glob('*/*.txt'))
    counts=Counter();empty=[];invalid=[];radii=[]
    for path in files:
        if not path.stat().st_size:
            empty.append(path.relative_to(root).as_posix());continue
        try:
            xyz=read_points(path)
            counts[len(xyz)]+=1
            radii.append(float(np.linalg.norm(xyz,axis=1).max()))
        except Exception as error:
            invalid.append({'file':path.relative_to(root).as_posix(),'error':str(error)})
    result={'root':str(root),'files':len(files),'point_counts':dict(counts),
            'empty_count':len(empty),'empty_files':empty,'invalid_count':len(invalid),'invalid_files':invalid}
    if radii:
        result['radius_min_median_max']=np.quantile(radii,[0,.5,1]).tolist()
    try:
        train=split_ids(root,'train');test=split_ids(root,'test')
        result.update(train_count=len(train),test_count=len(test),overlap=sorted(set(train)&set(test)),
                      missing_files=sorted({x for x in train+test if not (root/(x+'.txt')).is_file()}))
    except Exception as error:
        result['split_error']=str(error)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir',required=True)
    parser.add_argument('--output')
    args=parser.parse_args()
    result=audit(args.data_dir)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('empty_files','invalid_files','overlap','missing_files')},indent=2))
    print('split overlap:',len(result.get('overlap',[])),'missing files:',len(result.get('missing_files',[])))
    sys.exit(bool(result['empty_count'] or result['invalid_count'] or result.get('split_error') or result.get('overlap') or result.get('missing_files') or not result['files']))
