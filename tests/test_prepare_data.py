import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_data import export
from dataset_io import PointCloudIndex


class PrepareTests(unittest.TestCase):
    def args(self, source, output):
        return argparse.Namespace(dataset='MN40',class_names=None,output=str(output),
            input_npz=str(source),input_dir=None,train_npz=None,test_npz=None,
            train_h5=None,test_h5=None,num_point=4,sampling='none',normalization='none',
            seed=24,allow_pickle_data=False)

    def test_npz_export_roundtrip_and_split_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'data.npz';out=root/'output'
            train=np.arange(24,dtype=np.float32).reshape(2,4,3)
            np.savez(src,train_pc=train,train_label=np.array([8,8]),test_pc=train+100,test_label=np.array([8,8]))
            self.assertEqual(export(self.args(src,out)),{'train':2,'test':2})
            index=PointCloudIndex(out,'MN40','train',category='chair')
            np.testing.assert_allclose(index.points(0)[0],train[0])
            self.assertTrue((out/'chair/train_000000.txt').is_file())
            self.assertTrue((out/'chair/test_000000.txt').is_file())
            self.assertEqual(json.loads((out/'preprocessing.json').read_text())['counts']['test'],2)
            with self.assertRaises(FileExistsError):export(self.args(src,out))

    def test_overlap_leaves_no_partial_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'data.npz';out=root/'output'
            np.savez(src,train_pc=np.ones((1,4,3)),train_label=np.array([0]),train_id=np.array(['same']),
                     test_pc=np.ones((1,4,3)),test_label=np.array([0]),test_id=np.array(['same']))
            with self.assertRaisesRegex(ValueError,'overlap'):export(self.args(src,out))
            self.assertFalse(out.exists())

    def test_folder_export_preserves_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            (source/'chair').mkdir()
            for split in ('train','test'):
                np.savetxt(source/'chair'/(split+'.txt'),np.arange(12).reshape(4,3))
                (source/('shuffled_'+split+'_file_list.json')).write_text(json.dumps(['chair/'+split]))
            args=self.args(source,root/'output');args.input_npz=None;args.input_dir=str(source)
            export(args)
            self.assertEqual(len(PointCloudIndex(root/'output','MN40','test').records),1)

    def test_hdf5_export(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('Install h5py to test HDF5 conversion')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for split in ('train','test'):
                with h5py.File(root/(split+'.h5'),'w') as f:
                    f['data']=np.arange(12,dtype=np.float32).reshape(1,4,3)
                    f['label']=np.array([[8]],dtype=np.int64)
            args=self.args(root/'unused',root/'output');args.input_npz=None
            args.train_h5=[str(root/'train.h5')];args.test_h5=[str(root/'test.h5')]
            self.assertEqual(export(args),{'train':1,'test':1})


if __name__=='__main__':unittest.main()
