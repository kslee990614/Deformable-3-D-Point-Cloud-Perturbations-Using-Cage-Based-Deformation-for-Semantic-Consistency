"""Release I/O helpers; imports of heavy dependencies are deferred."""
from pathlib import Path

SYNSETS = ('02691156', '02773838', '02954340', '02958343', '03001627',
           '03261776', '03467517', '03624134', '03636649', '03642806',
           '03790512', '03797390', '03948459', '04099429', '04225987', '04379243')


def process_nested_file_ids(file_ids, base_dir):
    """Resolve every ID or fail; silently dropping labels corrupts batch alignment."""
    if isinstance(file_ids, str):
        file_ids = [file_ids]
    labels = []
    for item in file_ids:
        if isinstance(item, (list, tuple)):
            labels.extend(process_nested_file_ids(item, base_dir))
            continue
        matches = [i for i, synset in enumerate(SYNSETS)
                   if (Path(base_dir) / synset / (str(item) + '.txt')).is_file()]
        if len(matches) != 1:
            raise ValueError('Expected one category for {!r}; found {}'.format(item, len(matches)))
        labels.append(matches[0])
    return labels


def checkpoint_state(checkpoint):
    for key in ('states', 'state_dict', 'model_state_dict'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {key.removeprefix('module.') if hasattr(key, 'removeprefix')
            else (key[7:] if key.startswith('module.') else key): value
            for key, value in checkpoint.items()}


def load_cage_checkpoint(net, path):
    import torch
    if not path or not Path(path).is_file():
        raise FileNotFoundError('Provide a cage checkpoint with --ckpt: {}'.format(path))
    net.load_state_dict(checkpoint_state(torch.load(path, map_location='cpu')), strict=True)


def load_classifier(opt):
    import torch
    from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
    if not opt.classifier_ckpt or not Path(opt.classifier_ckpt).is_file():
        raise FileNotFoundError('Provide the victim checkpoint with --classifier_ckpt')
    constructors = {
        'dgcnn': lambda: DGCNN(1024, 20, output_channels=opt.num_classes),
        'pointnet': lambda: PointNetCls(k=opt.num_classes, feature_transform=False),
        'pointnet2': lambda: PointNet2ClsSsg(num_classes=opt.num_classes),
        'pointconv': lambda: PointConvDensityClsSsg(num_classes=opt.num_classes),
    }
    model = constructors[opt.classifier]().cuda()
    model.load_state_dict(checkpoint_state(torch.load(opt.classifier_ckpt, map_location='cpu')), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def attack_metrics(records):
    """Report the paper's all-input ASR and a clean-correct conditional ASR separately."""
    total = len(records)
    correct = sum(r['clean_prediction'] == r['label'] for r in records)
    successes = sum(r['adversarial_prediction'] != r['label'] for r in records)
    conditional = sum(r['clean_prediction'] == r['label'] and
                      r['adversarial_prediction'] != r['label'] for r in records)
    return {
        'num_samples': total,
        'clean_correct': correct,
        'adversarial_misclassified': successes,
        'clean_correct_attack_successes': conditional,
        'clean_accuracy_percent': 100 * correct / total if total else None,
        'asr_all_percent': 100 * successes / total if total else None,
        'asr_clean_correct_percent': 100 * conditional / correct if correct else None,
    }


def classification_labels(data, role, opt):
    key = role + "_class"
    if key in data:
        value = data[key]
        return value.tolist() if hasattr(value, "tolist") else list(value)
    return process_nested_file_ids(data[role + "_file"], opt.data_dir)
