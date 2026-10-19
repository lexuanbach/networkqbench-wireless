#!/usr/bin/env python3
"""Reconcile audit diagnostics from immutable campaign records."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from artifact_paths import ROOT, output_directory, portable
from revision_v121 import paired_interval

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'results')
    ap.add_argument('--output',type=Path,default=ROOT/'output/analysis/audit')
    a=ap.parse_args();out=output_directory(a.output);used=[]
    def read(name):
        p=a.input/name;used.append(p);return pd.read_csv(p)
    x=read('round4_20261019/template_round_instances.csv')
    components=['preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j','network_j','transition_j']
    pair=x.pivot(index=['boundary','size','seed','round'],columns='mode',values=components+['total_j'])
    delta=pd.DataFrame({c:pair[c]['template']-pair[c]['numeric'] for c in components+['total_j']}).reset_index()
    residual=float(abs(delta[components].sum(axis=1)-delta.total_j).max());assert residual<1e-7
    avg=delta.groupby(['boundary','size','seed'],as_index=False)[components+['total_j']].mean()
    rows=[]
    for boundary,g in avg.groupby('boundary'):
        for c in components+['total_j']:
            lo,hi=paired_interval(g,c);rows.append(dict(boundary=boundary,component=c,mean=g[c].mean(),low=lo,high=hi))
    avg.to_csv(out/'template_components_instances.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'template_components.csv',index=False)
    read('v1_2_2/sensitivity_summary.csv').to_csv(out/'sensitivity_summary.csv',index=False)
    read('round5_20261019/crossed_overheads.csv').to_csv(out/'crossed_overheads.csv',index=False)
    s=read('v1_2_2/sensitivity_all.csv');s[(s.scenario.str.startswith('Sleep')) & s.algorithm.isin(['keep_incumbent','local_search','onehot_structured'])].to_csv(out/'sleep_energy.csv',index=False)
    sleep=read('v1_2_1/sleep_consolidation.csv');sleep.groupby('workload').agg(cases=('seed','size'),empty=('optimum_sleep','mean'),consolidation=('consolidates','mean')).reset_index().to_csv(out/'sleep_consolidation.csv',index=False)
    grid=read('v1_2/all_on/sensitivity.csv');q=grid[~grid.algorithm.isin(['milp','heuristic','local_search']) & ~grid.algorithm.str.startswith('classical')]
    favorable=q[q.excess_j<0]
    marginal=q[q.boundary=='marginal'].groupby(['algorithm','shots']).agg(cells=('scenario_id','size'),favorable=('excess_j',lambda v:int((v<0).sum())),minimum_j=('excess_j','min')).reset_index()
    marginal.to_csv(out/'marginal_all_budgets.csv',index=False)
    p=read('v1_2/all_on/policies.csv')
    p['target_failed']=~p.training_targets_pass
    p.groupby('policy').agg(instances=('seed','size'),target_failures=('target_failed','sum'),invocations=('invoked_quantum','sum'),useful=('useful','mean'),applied=('accepted','mean')).reset_index().to_csv(out/'admission_targets.csv',index=False)
    noise=read('v1_2/noise/probabilities.csv');n=noise[noise.cx_depolarizing_parameter==0]
    n.groupby(['size','algorithm']).agg(instances=('seed','size'),cx=('cx','median'),depth=('depth','median'),shot_s=('shot_s','median'),max_error=('compilation_probability_error','max')).reset_index().to_csv(out/'noise_resources.csv',index=False)
    ins=read('v1_2/all_on/instances.csv');ins=ins[ins.seed>=15]
    path=a.input/'v1_2_2/diagnostics.json';used.append(path);d=json.loads(path.read_text())
    d.update(cooptimal_instances=int((ins.cooptimal_states>1).sum()),favorable_cells=len(favorable),favorable_scenarios=int(favorable.scenario_id.nunique()),best_grid_j=float(favorable.excess_j.min()),template_reconciliation_max_j=residual)
    assert d['cooptimal_instances']==47 and d['favorable_cells']==136 and d['favorable_scenarios']==68
    pd.DataFrame([dict(intensity_g_kwh=i,excess_j=118.05,carbon_g=i*118.05/3.6e6) for i in [50,400,800]]).to_csv(out/'carbon.csv',index=False)
    (out/'diagnostics.json').write_text(json.dumps(d,indent=2)+'\n')
    # Portable projections preserve historical metadata bytes and their original hashes.
    def normalize(value):
        if isinstance(value,dict):return {clean(k):normalize(v) for k,v in value.items()}
        if isinstance(value,list):return [normalize(v) for v in value]
        return clean(value) if isinstance(value,str) else value
    def clean(s):
        return 'artifact/'+s.split('/artifact/',1)[1] if '/artifact/' in s else s
    projections=[]
    for p in sorted(a.input.rglob('*manifest.json')):
        data=p.read_text()
        if chr(47)+'Users'+chr(47) in data:
            dest=out/'portable_manifests'/p.relative_to(a.input);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(normalize(json.loads(data)),indent=2)+'\n')
            projections.append(dict(source=portable(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),projection=dest.relative_to(out).as_posix()))
    (out/'manifest.json').write_text(json.dumps(dict(campaign='AuditDerived',new_measurements=False,bootstrap_resamples=20000,bootstrap_seed=20261017,unit='problem instance after averaging sampling repetitions and timing rounds',input_sha256={portable(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in used},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),portable_projections=projections),indent=2)+'\n')
    print('Audit diagnostics reconciled:',d['template_reconciliation_max_j'])
if __name__=='__main__':main()
