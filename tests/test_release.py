import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_utils import attack_metrics, checkpoint_state, process_nested_file_ids, SYNSETS


class ReleaseTests(unittest.TestCase):
    def test_asr_distinguishes_existing_clean_errors(self):
        rows = [dict(label=1, clean_prediction=1, adversarial_prediction=0),
                dict(label=1, clean_prediction=0, adversarial_prediction=0),
                dict(label=1, clean_prediction=1, adversarial_prediction=1)]
        result = attack_metrics(rows)
        self.assertAlmostEqual(result['asr_all_percent'], 200 / 3)
        self.assertEqual(result['asr_clean_correct_percent'], 50)
        self.assertIsNone(attack_metrics([])['asr_all_percent'])
        self.assertIsNone(attack_metrics(rows[1:2])['asr_clean_correct_percent'])

    def test_missing_or_ambiguous_labels_fail(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / SYNSETS[0]
            p.mkdir()
            (p/'shape.txt').touch()
            self.assertEqual(process_nested_file_ids([['shape']], root), [0])
            with self.assertRaises(ValueError):
                process_nested_file_ids(['shape', 'missing'], root)
            q = Path(root) / SYNSETS[1]
            q.mkdir()
            (q/'shape.txt').touch()
            with self.assertRaises(ValueError):
                process_nested_file_ids(['shape'], root)

    def test_checkpoint_prefix_only_removed_when_present(self):
        self.assertEqual(checkpoint_state({'states': {'module.weight': 1, 'bias': 2}}),
                         {'weight': 1, 'bias': 2})



    def test_core_parser_accepts_evaluation_paths(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; from option import BaseOptions; "
            "sys.argv=['core','--phase','adv_test','--ckpt','cage.pth',"
            "'--classifier_ckpt','victim.pth']; "
            "o=BaseOptions().parse(); assert o.phase=='adv_test'; assert o.classifier_ckpt=='victim.pth'"],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_selects_victim_class_count(self):
        for dataset, expected in [('MN40',40),('SCANOBJECTNN',15),('SHAPENET',16)]:
            result = subprocess.run([sys.executable, '-c',
                "import sys;from option import BaseOptions;sys.argv=['core','--dataset',sys.argv[1]];"
                "o=BaseOptions().parse();print(o.num_classes)",dataset],
                cwd=str(ROOT),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(int(result.stdout.strip()),expected)

    def test_all_shipped_python_parses(self):
        for path in ROOT.rglob('*.py'):
            with self.subTest(path=str(path)):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


if __name__ == '__main__':
    unittest.main()
