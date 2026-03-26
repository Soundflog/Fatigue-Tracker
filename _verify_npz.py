import numpy as np
for name in ['physionet_only', 'wsd4fedsrm_only', 'wrist_combined']:
    d = np.load(f'data/processed/{name}.npz', allow_pickle=True)
    doms = d['domains']
    hp = d['has_physio'].astype(bool)
    subj = d['pids']
    y = d['y']
    print(f"\n=== {name} ===")
    print(f"  X_imu: {d['X_imu'].shape}, X_physio: {d['X_physio'].shape}")
    print(f"  Domains: {list(np.unique(doms))}")
    print(f"  Subjects: {len(np.unique(subj))}")
    print(f"  has_physio: {hp.sum()}/{len(hp)}")
    print(f"  y balance: {y.mean():.1%} fatigue ({y.sum()}/{len(y)})")
