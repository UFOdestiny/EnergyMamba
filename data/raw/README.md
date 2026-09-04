# Raw energy-consumption arrays

Two of the public datasets used in the paper, carried over from the original
release. Both are hourly 2018 series in `(N, D, T)` layout with `D = 1`:

| File | Grid | Zones `N` | Steps `T` |
| --- | --- | --- | --- |
| `caiso.npy` | California ISO | 9 | 8783 |
| `nyiso.npy` | New York ISO | 11 | 8784 |

Convert either into the framework's on-disk format with `utils/generate.py`
(`--fmt NDT` matches the `(N, D, T)` layout; `--clip_neg` zeroes the few
negative CAISO readings):

```bash
python utils/generate.py --data_path data/raw/caiso.npy \
    --dataset caiso --years 2018 --fmt NDT --clip_neg

python utils/generate.py --data_path data/raw/nyiso.npy \
    --dataset nyiso --years 2018 --fmt NDT
```

Then place the zone adjacency matrix as an `(N, N)` `.npy` next to the dataset
folder and register both in `utils/registry.yaml`:

```yaml
caiso:
  data: caiso
  adj: caiso/caiso.npy
```

The grid-topology adjacency used in the paper is not redistributed here;
`utils/get_adj_mat.py` builds one from zone geometries.
