#!/usr/bin/env python3
"""Derive round-four evidence without changing historical inputs."""
import argparse, json, hashlib, platform
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from round4 import ROOT, DEFAULT_OUT, FAMILIES, sha
from analyze_v12 import per_instance, DEFAULT
from revision_v121 import paired_interval
from artifact_paths import portable, output_directory


def interval(g,col):
    lo,hi=paired_interval(g,col)
    return dict(mean=float(g[col].mean()),low=float(lo),high=float(hi),instances=len(g))


def static(a):
    s=pd.read_csv(a.previous/'scaling.csv')
    s['exact_gap_w']=(s.greedy_power_w-s.power_w).where(s.exact)
    s['bound_gap_w']=s.greedy_power_w-s.bound_w
    s['idle_w']=220*s.nodes;s['static_60s_bound_j']=60*s.bound_gap_w
    s.to_csv(a.output/'absolute_scaling_instances.csv',index=False)
    s.groupby(['nodes','functions']).agg(cases=('seed','size'),certified=('exact_gap_w','count'),
        exact_max_w=('exact_gap_w','max'),bounds=('bound_gap_w','count'),bound_max_w=('bound_gap_w','max'),
        idle_w=('idle_w','first'),static_60s_bound_j=('static_60s_bound_j','max')).reset_index().to_csv(a.output/'absolute_scaling_summary.csv',index=False)


def template(a):
    data=pd.read_csv(a.input/'all_on/outcomes.csv');data=data[data.seed>=15]
    data=data[((data.algorithm=='onehot_structured')&(data.shots==64))|data.algorithm.isin(['milp','heuristic'])]
    timing=pd.read_csv(a.third/'template_timing.csv')
    offline=pd.read_csv(a.third/'template_offline.csv').set_index(['size','round']).total_s
    records=[]
    for rnd,times in timing.groupby('round'):
        times=times.set_index(['size','seed'])
        for mode in ['template','numeric']:
            raw=data.copy()
            for (n,seed),ix in raw[raw.algorithm=='onehot_structured'].groupby(['size','seed']).groups.items():
                t=times.loc[(n,seed)]
                raw.loc[ix,'dense_preparation_s']=t.preparation_s
                raw.loc[ix,'circuit_build_s']=t.binding_s if mode=='template' else t.numeric_build_s
                raw.loc[ix,'compile_s']=0 if mode=='template' else t.numeric_compile_s
                raw.loc[ix,'shot_s']=t[f'{mode}_shot_s']
                if mode=='template':raw.loc[ix,'offline_s']+=offline.loc[(n,rnd)]
            for boundary in ['allocated','marginal']:
                p=per_instance(raw,replace(DEFAULT,boundary=boundary,marginal_fraction=.001))
                p=p[p.algorithm=='onehot_structured'].copy();p['round']=rnd;p['mode']=mode;p['boundary']=boundary;records.append(p)
    allp=pd.concat(records,ignore_index=True);allp.to_csv(a.output/'template_round_instances.csv',index=False)
    cols=['preparation_j','compute_j','total_j','excess_j','accepted','useful']
    pair=allp.pivot(index=['boundary','size','seed','round'],columns='mode',values=cols)
    differences=pd.DataFrame({c:pair[c]['template']-pair[c]['numeric'] for c in cols}).reset_index()
    differences.to_csv(a.output/'template_round_differences.csv',index=False)
    avg=differences.groupby(['boundary','size','seed'],as_index=False)[cols].mean();avg.to_csv(a.output/'template_paired_instances.csv',index=False)
    rows=[]
    for boundary,g in avg.groupby('boundary'):
        for c in cols:
            rounds=differences[differences.boundary==boundary].groupby('round')[c].mean()
            rows.append(dict(boundary=boundary,metric=c,**interval(g,c),round_min=rounds.min(),round_max=rounds.max()))
    pd.DataFrame(rows).to_csv(a.output/'template_paired_summary.csv',index=False)


def transfer(a):
    paths=sorted((a.campaign/'transfer').glob('*/outcomes.csv'))
    assert len(paths)==16,'incomplete transfer campaign'
    raw=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    probs=pd.concat([pd.read_csv(p.parent/'probabilities.csv') for p in paths],ignore_index=True)
    resources=pd.concat([pd.read_csv(p.parent/'resources.csv') for p in paths],ignore_index=True)
    resources.to_csv(a.output/'transfer_resources.csv',index=False)
    probs['below_uniform']=probs.p_quality < probs.uniform-1e-12
    probs.to_csv(a.output/'transfer_probabilities.csv',index=False)
    probs.groupby(['workload','size','method']).agg(p_quality=('p_quality','mean'),uniform=('uniform','mean'),below=('below_uniform','sum'),instances=('seed','size')).reset_index().to_csv(a.output/'transfer_probability_summary.csv',index=False)
    instances=[]
    for family,g in raw.groupby('workload'):
        old=pd.read_csv(a.input/family/'outcomes.csv');old=old[old.seed>=15]
        classical=old[old.algorithm.isin(['milp','heuristic'])]
        prof=old[old.algorithm=='onehot_structured'].drop_duplicates(['size','seed','shots']).set_index(['size','seed','shots'])
        for profile in ['measured','common']:
            q=g.copy()
            if profile=='common':
                keys=list(zip(q['size'],q.seed,q.shots))
                for c in ['dense_preparation_s','circuit_build_s','compile_s','postprocessing_s','offline_s','shot_s','evals','local_s','lookup_s']:
                    q[c]=prof.loc[keys,c].to_numpy()
            for boundary in ['allocated','marginal']:
                pi=per_instance(pd.concat([classical,q],ignore_index=True),replace(DEFAULT,boundary=boundary,marginal_fraction=.001))
                pi=pi[~pi.algorithm.isin(['milp','heuristic'])].copy();pi['profile']=profile;pi['boundary']=boundary;instances.append(pi)
    pi=pd.concat(instances,ignore_index=True);pi.to_csv(a.output/'transfer_instances.csv',index=False)
    sums=[];pairs=[]
    for (family,profile,boundary,alg,shots),g in pi.groupby(['workload','profile','boundary','algorithm','shots']):
        stat=interval(g,'excess_j')
        sums.append(dict(workload=family,profile=profile,boundary=boundary,algorithm=alg,shots=shots,
                         **stat,quality=g.quality.mean(),useful=g.useful.mean(),accepted=g.accepted.mean(),
                         compute_j=g.compute_j.mean(),offline_j=g.offline_j.mean(),p_quality=g.p_quality.mean(),wins=int((g.excess_j< -1e-8).sum())))
    for (family,profile,boundary,shots),g in pi.groupby(['workload','profile','boundary','shots']):
        for objective in ['expectation','useful_mass']:
            d=g[g.algorithm=='direct_'+objective].set_index(['size','seed'])
            c=g[g.algorithm=='circular_'+objective].set_index(['size','seed']).loc[d.index]
            diff=(d.total_j-c.total_j).rename('difference_j').reset_index()
            pairs.append(dict(workload=family,profile=profile,boundary=boundary,shots=shots,objective=objective,**interval(diff,'difference_j'),
                              useful_difference=float((d.useful-c.useful).mean()),accepted_difference=float((d.accepted-c.accepted).mean()),
                              wins=int((diff.difference_j< -1e-8).sum())))
    pd.DataFrame(sums).to_csv(a.output/'transfer_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(a.output/'transfer_paired_summary.csv',index=False)


def scaling(a):
    runs=pd.read_csv(a.campaign/'deadline_runs.csv',dtype={'solution':str,'applied_action':str})
    assert len(runs)==600 and len(runs[['nodes','functions','seed']].drop_duplicates())==100
    old=pd.read_csv(a.previous/'scaling.csv');new=pd.read_csv(a.campaign/'new_references.csv')
    refs=pd.concat([old.assign(campaign='Follow-up'),new.assign(campaign='Round4DeadlineReference')],ignore_index=True)
    assert len(refs)==100
    refs.to_csv(a.output/'scaling_references.csv',index=False)
    runs=runs.merge(refs[['nodes','functions','seed','exact','power_w','bound_w']].rename(columns={'exact':'reference_exact','power_w':'reference_power_w','bound_w':'reference_bound_w'}),on=['nodes','functions','seed'],validate='many_to_one')
    runs['exact_gap_w']=(runs.power_w-runs.reference_power_w).where(runs.reference_exact & runs.feasible)
    runs['bound_gap_w']=(runs.power_w-runs.reference_bound_w).where(runs.feasible)
    runs['keep_j']=60*runs.incumbent_w;runs['minus_keep_j']=runs.total_j-runs.keep_j
    base=runs[runs.algorithm=='heuristic'].set_index(['nodes','functions','seed','deadline_s'])
    keys=list(zip(runs.nodes,runs.functions,runs.seed,runs.deadline_s))
    runs['minus_greedy_j']=runs.total_j.to_numpy()-base.loc[keys,'total_j'].to_numpy()
    runs['feasible_timely']=runs.feasible & runs.timely
    runs.to_csv(a.output/'deadline_instances.csv',index=False)
    rows=[]
    for (n,f,d,alg),g in runs.groupby(['nodes','functions','deadline_s','algorithm']):
        # Each configuration is a stratum of ten fixed problem identities; deadlines are not pooled.
        h=g.copy();h['size']=n*f
        keep=interval(h,'minus_keep_j');greedy=interval(h,'minus_greedy_j')
        rows.append(dict(nodes=n,functions=f,deadline_s=d,algorithm=alg,cases=len(g),feasible=int(g.feasible.sum()),
                         feasible_timely=int(g.feasible_timely.sum()),accepted=int(g.accepted.sum()),
                         solver_called=int(g.solver_called.sum()),median_build_s=g.construction_s.median(),
                         median_solve_s=g.solve_s.median(),median_total_s=g.runtime_s.median(),max_total_s=g.runtime_s.max(),
                         mean_compute_j=g.compute_j.mean(),utilization=g.utilization.mean(),
                         exact_count=int(g.exact_gap_w.notna().sum()),exact_max_w=g.exact_gap_w.max(),
                         bound_count=int(g.bound_gap_w.notna().sum()),bound_max_w=g.bound_gap_w.max(),
                         minus_keep_j=keep['mean'],keep_low=keep['low'],keep_high=keep['high'],
                         minus_greedy_j=greedy['mean'],greedy_low=greedy['low'],greedy_high=greedy['high']))
    pd.DataFrame(rows).to_csv(a.output/'deadline_summary.csv',index=False)
    refsum=refs.groupby(['nodes','functions']).agg(cases=('seed','size'),certified=('exact','sum'),feasible=('feasible','sum'),
            bounds=('bound_w','count'),max_gap=('mip_gap','max'),median_s=('runtime_s','median')).reset_index()
    refsum.to_csv(a.output/'reference_summary.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'results/v1_2');ap.add_argument('--previous',type=Path,default=ROOT/'results/v1_2_1')
    ap.add_argument('--third',type=Path,default=ROOT/'results/v1_2_2');ap.add_argument('--campaign',type=Path,default=DEFAULT_OUT);ap.add_argument('--output',type=Path,default=ROOT/'output/analysis/round4')
    ap.add_argument('--stages',nargs='+',choices=['static','template','transfer','scaling'],default=['static','template','transfer','scaling'])
    a=ap.parse_args();a.output=output_directory(a.output)
    for stage in a.stages:globals()[stage](a);print('analysis',stage,flush=True)
    (a.output/'analysis_manifest.json').write_text(json.dumps(dict(campaign="Round4Derived", generated_utc=datetime.now(timezone.utc).isoformat(), environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__), input_sha256={portable(p):sha(p) for p in [a.previous/"scaling.csv",a.third/"template_timing.csv",a.third/"template_offline.csv"]+list(a.input.glob("*/outcomes.csv"))+list((a.campaign/"transfer").glob("*/*.csv"))+[a.campaign/"deadline_runs.csv",a.campaign/"new_references.csv"] if p.exists()}, bootstrap_resamples=20000,bootstrap_seed=20261017,
        unit='problem instance, size-stratified for physical families; ten seeds separately within each scaling configuration',
        timing='round-specific event replay before averaging rounds within instances',
        source_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'round4.py',ROOT/'analyze_v12.py',ROOT/'revision_v121.py']}),indent=2)+'\n')
if __name__=='__main__':main()
