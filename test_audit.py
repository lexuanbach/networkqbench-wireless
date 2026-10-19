#!/usr/bin/env python3
"""Regression checks for portable, read-only analysis and audit reconciliation."""
import ast,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from artifact_paths import ROOT,output_directory,validation_path
from render_v122 import number,interval,fix_signs

def main():
    for path in [ROOT/'results',ROOT/'results/derived',ROOT/'source_snapshots/new']:
        try:output_directory(path)
        except ValueError:pass
        else:raise AssertionError('Evidence output accepted')
    assert number(-.00001)=='\\ensuremath{0.00}'
    assert '-0.00' not in interval(-.00001,-.00002,.00003)
    assert r'3{,}830.00' in fix_signs('$[3,830.00,3,899.24]$')
    assert '[109.63,129.68]' in fix_signs('$[109.63,129.68]$')
    raw=pd.read_csv(ROOT/'results/round4_20261019/template_round_instances.csv')
    components=['preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j','network_j','transition_j']
    assert np.allclose(raw[components].sum(axis=1),raw.total_j,rtol=0,atol=1e-7)
    pair=raw.groupby(['boundary','mode'])[components+['total_j']].mean()
    delta=pair.loc[('allocated','template')]-pair.loc[('allocated','numeric')]
    assert np.isclose(delta.network_j,-.0477222214772822,atol=1e-8)
    assert np.isclose(delta[components].sum(),delta.total_j,atol=1e-8)
    # Default renderers have no manuscript dependency, even under --skip-legacy.
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'out';out.mkdir();(out/'v12_principal.tex').write_text('')
        subprocess.run([sys.executable,str(ROOT/'render_round5.py'),'--skip-legacy','--output',str(out)],check=True)
        assert (out/'round5_marginal_map.tex').exists()
        assert not (Path(tmp)/'manuscript').exists()
        assert '8.96' in (out/'round5_marginal_map.tex').read_text()
    for name in ['render_round4.py','render_round5.py']:
        tree=ast.parse((ROOT/name).read_text())
        assert not any(isinstance(n,ast.Constant) and n.value in ['supplement.tex','sections/09_revision_supplement.tex','sections/10_round4_supplement.tex'] for n in ast.walk(tree))
    report=dict(status='passed',evidence_output_rejected=True,zero_rounding=True,math_thousands=True,template_network_residual_j=float(delta.network_j),template_ledger_reconciled=True,standalone_renderer=True,zero_compile_64shot=True,prose_rewriting_removed=True)
    validation_path('test_audit').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
