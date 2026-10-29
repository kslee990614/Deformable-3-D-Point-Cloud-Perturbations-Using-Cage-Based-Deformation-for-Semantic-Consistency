"""Read-only checks for the legacy Linux/CUDA runtime."""
import importlib
import platform
import sys

print('Python:', sys.version)
print('Platform:', platform.platform())
failures = []
if platform.system() != 'Linux':
    failures.append('The custom CUDA extensions and PyMesh runtime target Linux')
for name in ('torch', 'numpy', 'scipy', 'faiss', 'openmesh', 'pymesh', 'plyfile',
             'matplotlib', 'joblib', 'tqdm', 'pytorch_points.network.geo_operations'):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, '__version__', 'import OK'))
    except Exception as exc:
        failures.append('{}: {}'.format(name, exc))
try:
    import torch
    print('PyTorch CUDA:', torch.version.cuda)
    if not torch.cuda.is_available():
        failures.append('CUDA GPU unavailable')
except ImportError:
    pass
for failure in failures:
    print('FAIL:', failure)
sys.exit(bool(failures))
