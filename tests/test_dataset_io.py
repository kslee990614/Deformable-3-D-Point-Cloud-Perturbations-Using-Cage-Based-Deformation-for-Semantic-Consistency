import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_io import PointCloudIndex, prepare_points, validate_disjoint


class UnifiedDataTests(unittest.TestCase):
    def test_identity_and_explicit_normalization(self):
        x=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],dtype=np.float32)
        np.testing.assert_array_equal(prepare_points(x,4),x)
        y=prepare_points(x+10,4,normalization='unit_sphere')
        np.testing.assert_allclose(y.mean(0),0,atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(y,axis=1).max()),1)
        z=prepare_points(x,4,normalization='bbox_ground')
        self.assertEqual(float(z[:,1].min()),-1)

    def test_fps_is_repeatable_and_not_implicit(self):
        x=np.random.RandomState(4).randn(30,3)
        np.testing.assert_array_equal(prepare_points(x,8,'fps',seed=6),prepare_points(x,8,'fps',seed=6))
        self.assertEqual(len(np.unique(prepare_points(x,8,'fps'),axis=0)),8)
        with self.assertRaises(ValueError):prepare_points(x,8)
        with self.assertRaises(ValueError):prepare_points(x[:2],8,'fps')

    def make_folder(self,root,overlap=False):
        (root/'airplane').mkdir()
        for name in ('a','b','c','d'):
            np.savetxt(root/'airplane'/(name+'.txt'),np.eye(3))
        (root/'shuffled_train_file_list.json').write_text(json.dumps(['shape_data/airplane/a','shape_data/airplane/b']))
        test=['shape_data/airplane/a','shape_data/airplane/c'] if overlap else ['shape_data/airplane/c','shape_data/airplane/d']
        (root/'shuffled_test_file_list.json').write_text(json.dumps(test))

    def test_folder_class_order_and_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_folder(root)
            data=PointCloudIndex(root,'MN40','train',category='airplane')
            self.assertEqual(len(data.records),2)
            self.assertEqual(data.points(0)[1],0)
            self.assertNotIn('airplane/c',[r[0] for r in data.records])

    def test_train_test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_folder(root,True)
            with self.assertRaisesRegex(ValueError,'leakage'):validate_disjoint(root)

    def test_npz_numeric_labels_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.npz'
            np.savez(path,train_pc=np.ones((2,4,3)),train_label=np.array([8,8]))
            data=PointCloudIndex(path,'MN40','train',category='chair')
            self.assertEqual([r[1] for r in data.records],[8,8])

    def test_scan_folder_needs_classifier_label_order(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'class_names'):
                PointCloudIndex(temp,'SCANOBJECTNN','train')

    def test_npz_explicit_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.npz'
            np.savez(path,train_pc=np.ones((2,4,3)),train_label=np.array([8,8]),
                     train_id=np.array(['a','b']),test_id=np.array(['a','c']))
            with self.assertRaisesRegex(ValueError,'leakage'):
                PointCloudIndex(path,'MN40','train')


if __name__=='__main__':unittest.main()
