#!/usr/bin/env python3
"""Inventory, verify, or export the local archive, including Git-ignored data."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import tarfile

SKIP={'.git','.venv','.pixi','node_modules','__pycache__','.cache','.pytest_cache',
      '.ipynb_checkpoints','.rsync-partial','runtime-cache','numba-cache'}


def root_path():
    for p in Path(__file__).resolve().parents:
        if (p/'provenance/hpc_download_plan.tsv').is_file():return p
    raise RuntimeError('Cannot locate the archive root')


def files(root):
    for folder,directories,names in os.walk(root):
        directories[:]=[d for d in directories if d not in SKIP and not (Path(folder)/d).is_symlink()]
        for name in sorted(names):
            p=Path(folder)/name
            if name=='.DS_Store' or name.startswith('~$') or name.endswith('.partial') or p.is_symlink():continue
            if p.relative_to(root).as_posix()=='provenance/files.tsv':continue
            yield p


def inventory(root):
    rows=[]
    for p in files(root):
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        rows.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest))
    dest=root/'provenance/files.tsv'
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],delimiter='\t')
        writer.writeheader();writer.writerows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['inventory','verify','export'])
    parser.add_argument('--output',type=Path,help='Destination .tar.gz file for export')
    args=parser.parse_args();root=root_path()
    if args.operation=='inventory':
        rows=inventory(root)
        print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),manifest='provenance/files.tsv'),indent=2))
        return 0
    if args.operation=='verify':
        with (root/'provenance/files.tsv').open(newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
        failures=[]
        for row in rows:
            p=root/row['path']
            if not p.is_file() or p.stat().st_size!=int(row['bytes']):failures.append(row['path']);continue
            with p.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=row['sha256']:failures.append(row['path'])
        print(json.dumps(dict(checked=len(rows),failures=failures),indent=2))
        return bool(failures)
    if args.output is None:parser.error('--output is required for export')
    output=args.output.expanduser().resolve()
    if output==root or root in output.parents:parser.error('Write the export outside the repository')
    if output.exists():parser.error('Refusing to overwrite '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=inventory(root)
    temporary=output.with_name(output.name+'.partial')
    if temporary.exists():parser.error('A partial export exists: '+str(temporary))
    with tarfile.open(temporary,'w:gz',compresslevel=6) as archive:
        for p in files(root):archive.add(p,arcname='conformer-encoding/'+p.relative_to(root).as_posix(),recursive=False)
        archive.add(root/'provenance/files.tsv',arcname='conformer-encoding/provenance/files.tsv',recursive=False)
    temporary.rename(output)
    with output.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    output.with_name(output.name+'.sha256').write_text(digest+'  '+output.name+'\n')
    print(json.dumps(dict(archive=str(output),files=len(rows)+1,bytes=output.stat().st_size,sha256=digest),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
