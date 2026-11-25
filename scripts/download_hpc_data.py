#!/usr/bin/env python3
"""Resume direct SSH/rsync collection of the paper's archived HPC materials.

Requires Python 3.11+, ssh and rsync. It never submits jobs, deletes remote files,
or overwrites existing local files. Paths and dataset groups are in the TSV plan.
"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import datetime
import gzip
import json
import re
import shlex
import shutil
import subprocess
import sys


def archive_root():
    here = Path(__file__).resolve()
    for root in here.parents:
        if (root / 'provenance/hpc_download_plan.tsv').is_file():
            return root
    raise RuntimeError('Run the script from the completed reproduction repository')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entropy-host', help='SSH target, for example maboyer@chem-entr-c01.artsci.tamu.edu')
    parser.add_argument('--grace-host', help='Your Grace SSH target or configured SSH alias')
    parser.add_argument('--entropy-root', default='/scratch/user/maboyer/llm')
    parser.add_argument('--grace-root', default='/scratch/user/maboyer/llm/encoding')
    parser.add_argument('--entropy-port', type=int, default=22)
    parser.add_argument('--grace-port', type=int, default=22)
    parser.add_argument('--cluster', choices=['all','entropy','grace'], default='all')
    parser.add_argument('--only', nargs='+', help='Download only these dataset IDs')
    parser.add_argument('--dry-run', action='store_true', help='List planned transfers without writing files')
    parser.add_argument('--bandwidth-kib', type=int, help='Optional rsync bandwidth limit in KiB/second')
    parser.add_argument('--keep-smiles-compressed', action='store_true', help='Do not expand geom_drugs.smi.gz after its transfer')
    args = parser.parse_args()
    root = archive_root()
    with (root/'provenance/hpc_download_plan.tsv').open(newline='') as f:
        plan = list(csv.DictReader(f, delimiter='\t'))
    if args.only:
        unknown=set(args.only)-{row['id'] for row in plan}
        if unknown:parser.error('Unknown dataset IDs: '+', '.join(sorted(unknown)))
        plan=[row for row in plan if row['id'] in args.only]
    if args.cluster!='all':plan=[row for row in plan if row['cluster']==args.cluster]
    if not plan:parser.error('No datasets selected')
    if not shutil.which('rsync') or not shutil.which('ssh'):
        parser.error('Both ssh and rsync must be installed')
    for cluster in {row['cluster'] for row in plan}:
        host=getattr(args,cluster+'_host')
        if not host:parser.error('--'+cluster+'-host is required for the selected transfers')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@:-]*',host):
            parser.error('Invalid SSH target: '+host)
    receipts=[]
    interrupted=False
    try:
        for index,row in enumerate(plan,1):
            cluster=row['cluster'];host=getattr(args,cluster+'_host')
            base=getattr(args,cluster+'_root');port=getattr(args,cluster+'_port')
            relative=PurePosixPath(row['remote'])
            local=PurePosixPath(row['local'])
            if relative.is_absolute() or local.is_absolute() or '..' in relative.parts or '..' in local.parts:
                raise ValueError('Plan paths must stay inside their declared roots')
            remote=str(PurePosixPath(base)/relative)
            dest=root/str(local)
            # A file goes into its destination directory; a directory preserves its contents.
            source=remote+'/' if row['kind']=='directory' else remote
            target=dest if row['kind']=='directory' else dest.parent
            ssh=['ssh','-p',str(port),'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
            command=['rsync','-aL','--ignore-existing','--partial','--partial-dir=.rsync-partial',
                     '--progress','--human-readable',
                     '--exclude=.git/','--exclude=.venv/','--exclude=node_modules/',
                     '--exclude=environment/','--exclude=env/','--exclude=.pixi/',
                     '--exclude=.cache/','--exclude=.matplotlib/','--exclude=.ipython/',
                     '--exclude=.jupyter_runtime/','--exclude=runtime-cache/',
                     '--exclude=__pycache__/','--exclude=.pytest_cache/',
                     '--exclude=*.pyc','--exclude=*.tmp','--exclude=materialization.lock','--exclude=merge.lock',
                     '--exclude=hpc-rest-*.out','--exclude=numba-cache/',
                     '--exclude=.numba-cache/','--exclude=torchinductor*/',
                     '-e',shlex.join(ssh)]
            if args.bandwidth_kib:command.append('--bwlimit='+str(args.bandwidth_kib))
            command += ['--',host+':'+shlex.quote(source),str(target)+'/']
            print(f'\n[{index}/{len(plan)}] {row["id"]}: {row["description"]}',flush=True)
            print(shlex.join(command),flush=True)
            rec=dict(id=row['id'],cluster=cluster,remote=remote,local=str(local),
                     started=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if args.dry_run:
                rec['status']='planned';receipts.append(rec);continue
            # Check existence separately so a missing frozen input is explicit in the report.
            probe=subprocess.run(ssh+[host,'test -e '+shlex.quote(remote)])
            if probe.returncode:
                rec.update(status='missing_or_unreachable',returncode=probe.returncode)
                receipts.append(rec)
                print('SOURCE UNAVAILABLE: '+remote,file=sys.stderr,flush=True)
                continue
            target.mkdir(parents=True,exist_ok=True)
            result=subprocess.run(command)
            rec.update(status='copied' if result.returncode==0 else 'failed',returncode=result.returncode,
                       finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
            receipts.append(rec)
            if result.returncode==0 and row['id']=='geom-smiles-main' and not args.keep_smiles_compressed:
                compressed=dest
                main_file=dest.with_suffix('')
                if not main_file.exists():
                    temporary=main_file.with_name(main_file.name+'.partial')
                    print('Expanding the compact GEOM SMILES database locally: '+str(main_file),flush=True)
                    with gzip.open(compressed,'rb') as source_stream, temporary.open('wb') as destination_stream:
                        shutil.copyfileobj(source_stream,destination_stream,1024*1024)
                    temporary.rename(main_file)
    except KeyboardInterrupt:
        interrupted=True
        print('\nInterrupted. Re-run the same command to resume partial transfers.',file=sys.stderr)
    if not args.dry_run:
        folder=root/'data/transfer_receipts';folder.mkdir(parents=True,exist_ok=True)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (folder/(stamp+'.json')).write_text(json.dumps(receipts,indent=2)+'\n')
    failures=[r for r in receipts if r['status'] not in {'copied','planned'}]
    print(f'\nFinished entries: {len(receipts)}/{len(plan)}; unavailable or failed: {len(failures)}',flush=True)
    if failures:
        print('Retry IDs: '+' '.join(row['id'] for row in failures),file=sys.stderr)
    if interrupted:return 130
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
