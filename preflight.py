#!/usr/bin/env python3
"""Check the recorded Python environment and external rendering tools."""
import importlib.metadata, json, platform, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    versions={};errors=[]
    expected=(ROOT/'.python-version').read_text().strip()
    if platform.python_version()!=expected:errors.append(f'Python {expected} required for the recorded environment; found {platform.python_version()}')
    for file in ['requirements-energy.txt','requirements-validation.txt']:
        for line in (ROOT/file).read_text().splitlines():
            if '==' not in line or line.startswith('#'):continue
            package,wanted=line.strip().split('==')
            try:found=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:found='missing'
            versions[package]=found
            if found!=wanted:errors.append(f'{package}: expected {wanted}, found {found}')
    external={}
    for name in ['latex','pdflatex','latexmk','dvipng','kpsewhich','pdftoppm','pdffonts','pdftotext','pdfinfo']:
        path=shutil.which(name);external[name]=bool(path)
        if not path:errors.append(name+' is required')
    if external['kpsewhich']:
        for name in ['txfonts.sty','type1cm.sty','type1ec.sty','elsarticle.cls','xr.sty']:
            ok=subprocess.run(['kpsewhich',name],capture_output=True,text=True).returncode==0
            external[name]=ok
            if not ok:errors.append('TeX resource missing: '+name)
    tex=subprocess.run(['pdflatex','--version'],capture_output=True,text=True).stdout.splitlines()[0] if external['pdflatex'] else 'missing'
    report=dict(python=platform.python_version(),packages=versions,external=external,tex_version=tex,errors=errors,status='passed' if not errors else 'failed')
    print(json.dumps(report,indent=2));return bool(errors)
if __name__=='__main__':sys.exit(main())
