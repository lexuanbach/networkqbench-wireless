#!/usr/bin/env python3
"""Export explicit versioned outputs; default to the unified 1.2 benchmark."""
from pathlib import Path
import argparse, shutil, subprocess, sys

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manuscript', type=Path, required=True)
    ap.add_argument('--version', choices=['1.1','1.2'], default='1.2')
    args=ap.parse_args(); root=Path(__file__).resolve().parent
    if args.version=='1.2':
        subprocess.run([sys.executable,str(root/'render_v12.py'),'--manuscript',str(args.manuscript.resolve())],check=True)
        return
    source=root/'output/sustainability'
    for sub,names in {'figures':['accounting.pdf','energy_quality.pdf','joint_frontier.pdf','net_energy.pdf','admission.pdf'],
                      'tables':['energy_numbers.tex','principal_results.tex']}.items():
        (args.manuscript/sub).mkdir(parents=True,exist_ok=True)
        for name in names:shutil.copy2(source/name,args.manuscript/sub/name)
if __name__=='__main__':main()
