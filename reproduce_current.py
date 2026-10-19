#!/usr/bin/env python3
"""Regenerate the current local evidence and presentation from stored campaigns.

No measurements are rerun. No manuscript tree is required. --manuscript copies
only generated tables and figures; prose and bibliography are never rewritten.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from artifact_paths import ROOT,output_directory,portable

def hashes():
    return {portable(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in ['results','source_snapshots'] for p in sorted((ROOT/base).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'output/current');ap.add_argument('--manuscript',type=Path);ap.add_argument('--validate',action='store_true');a=ap.parse_args();o=output_directory(a.output);before=hashes();steps=[]
    def run(script,*args):
        result=subprocess.run([sys.executable,str(ROOT/(script+'.py')),*map(str,args)],cwd=ROOT,capture_output=True,text=True)
        (o/(script+'.log')).write_text(result.stdout+result.stderr)
        steps.append(dict(script=script,exit_code=result.returncode));print(script,result.returncode,flush=True)
        if result.returncode:raise RuntimeError('See '+str(o/(script+'.log')))
    run('preflight')
    run('analyze_v121','--output',o/'analysis/v121')
    run('analyze_v122','--output',o/'analysis/v122')
    run('analyze_round4','--output',o/'analysis/round4')
    run('analyze_round5','--output',o/'analysis/round5')
    run('analyze_audit','--output',o/'analysis/audit')
    args=['--input',o/'analysis/round5','--fourth',o/'analysis/round4','--output',o]
    if a.manuscript:args+=['--manuscript',a.manuscript.resolve()]
    run('render_round5',*args)
    args=['--input',o/'analysis/audit','--output',o]
    if a.manuscript:args+=['--manuscript',a.manuscript.resolve()]
    run('render_audit',*args)
    if a.validate:
        for script in ['check_artifact','test_benchmark','test_energy','test_v12','validate_v12','test_v121','test_v122','test_round4','test_round5','validate_energy_results','test_audit']:
            run(script)
    after=hashes();assert before==after,'Frozen evidence changed during reproduction'
    (o/'reproduction.json').write_text(json.dumps(dict(status='passed',steps=steps,preserved_files=len(before),input_sha256=before,publication='local only; public v1.2.2 unchanged'),indent=2)+'\n')
if __name__=='__main__':main()
