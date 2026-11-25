#!/usr/bin/env python3
"""Recover the original coordinate samples using the native McUtils sampler.

Run inside a compute allocation with read access to the original combined.zarr.
This reads the entire coordinate stream. It does not submit a job or overwrite
existing samples, and a failed original checksum remains explicitly unverified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

SAMPLES = {
    'angle': dict(code=-1, name='geom_angles_27dc458757bb9fb8f6b7.npy', sha256=None),
    'dihedral': dict(code=-2, name='geom_dihedrals_3b48e9d569153cd1e32c.npy',
                     sha256='f46880cf26bcf58f36d2e57ef448b83d058e0ce2253455a4aaede0da2abee5eb'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr', required=True, type=Path)
    parser.add_argument('--kind', choices=['angle', 'dihedral', 'both'], default='both')
    args = parser.parse_args()
    root = next(p for p in Path(__file__).resolve().parents
                if (p / 'provenance/hpc_download_plan.tsv').is_file())
    native = root / 'workflows/geom-quantized-local-acceptance-20261123/candidate'
    sys.path.insert(0, str(native))
    import numpy as np
    import zarr
    from McUtils.Zachary.FittableModels import sample_masked_values
    group = zarr.open_group(str(args.zarr), mode='r')
    output = root / 'data/recovered_coordinate_samples'
    output.mkdir(parents=True, exist_ok=True)
    failed = False
    for kind in SAMPLES if args.kind == 'both' else [args.kind]:
        settings = SAMPLES[kind]
        path = output / settings['name']
        candidate = path.with_name(path.stem + '.candidate.npy')
        metadata = path.with_suffix('.json')
        if any(p.exists() for p in (path, candidate, metadata)):
            raise FileExistsError('Recovery output already exists for ' + kind)
        start = time.monotonic()
        last = [start]

        def mask(data, sl):
            now = time.monotonic()
            if now - last[0] >= 30:
                print(kind, f'{sl.stop / data["vals"].shape[0]:.1%}', flush=True)
                last[0] = now
            return np.asarray(data['types'][sl]) == settings['code']

        values = sample_masked_values(group, 10_000_000, mask_fn=mask,
                                      chunk_size=10_000_000, seed=20261007)
        maximum = np.pi if kind == 'angle' else 2 * np.pi
        if values.shape != (10_000_000,) or not np.all(np.isfinite(values)) or not np.all((values >= 0) & (values <= maximum)):
            raise ValueError('Recovered sample fails original shape/range checks')
        with candidate.open('xb') as f:
            np.save(f, values)
        with candidate.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        expected = settings['sha256']
        identical = digest == expected if expected is not None else None
        if identical:
            candidate.rename(path)
        failed |= identical is False
        report = dict(kind=kind, status='original_checksum_match' if identical else
                      'checksum_mismatch' if identical is False else 'regenerated_original_checksum_unavailable',
                      sha256=digest, expected_original_sha256=expected,
                      file=str((path if identical else candidate).relative_to(root)),
                      zarr_path=str(args.zarr.resolve()), vals_shape=list(group['vals'].shape),
                      sample_size=10_000_000, seed=20261007, chunk_size=10_000_000,
                      type_code=settings['code'], units='radians', numpy_version=np.__version__,
                      sampler='McUtils.Zachary.FittableModels.sample_masked_values',
                      elapsed_seconds=time.monotonic() - start)
        metadata.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
