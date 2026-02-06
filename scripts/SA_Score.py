import os, sys, importlib.util, gzip, pickle

here = os.path.dirname(__file__)
target = os.path.join(here, 'sascorer.py')
spec = importlib.util.spec_from_file_location('sascorer', target)
mod = importlib.util.module_from_spec(spec)
sys.modules['sascorer'] = mod
spec.loader.exec_module(mod)

try:
    with gzip.open(os.path.join(here, 'fpscores.pkl.gz'), 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            fs = data
        elif isinstance(data, (list, tuple)):
            try:
                fs = {k: v for (k, v) in data}
            except Exception:
                fs = {v: k for (k, v) in data}
        else:
            fs = None
        if fs is not None:
            mod._fscores = fs
except Exception as e:
    print("⚠️ Failed to load fpscores.pkl.gz:", e)

class sascorer:
    calculateScore = staticmethod(mod.calculateScore)
