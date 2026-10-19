#!/usr/bin/env python3
"""Derive paired revision tables from frozen inputs and separately measured additions."""
import argparse,json
from artifact_paths import output_directory
from dataclasses import replace
import numpy as np,pandas as pd
from revision_v121 import *


def compare(frame):
    rows=[]
    base=frame[frame.algorithm=='milp'].set_index(['size','seed'])
    greedy=frame[frame.algorithm=='heuristic'].set_index(['size','seed'])
    for (alg,shots),g in frame.groupby(['algorithm','shots']):
        g=g.copy();keys=list(zip(g['size'],g.seed))
        g['excess_j']=g.total_j.to_numpy()-base.loc[keys,'total_j'].to_numpy()
        g['greedy_excess_j']=g.total_j.to_numpy()-greedy.loc[keys,'total_j'].to_numpy()
        row=dict(algorithm=alg,shots=shots,**{c:float(g[c].mean()) for c in g.select_dtypes('number').columns if c not in ['size','seed','shots']})
        row['p_quality_median']=float(g.p_quality.median()) if 'p_quality' in g else None
        row['ci_low_j'],row['ci_high_j']=paired_interval(g,'excess_j')
        row['greedy_low_j'],row['greedy_high_j']=paired_interval(g,'greedy_excess_j')
        row.update(wins=int((g.excess_j< -1e-8).sum()),instances=len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def static(out):
    ins=pd.read_csv(FROZEN/'all_on/instances.csv');ins=ins[ins.seed>=15]
    pi=pd.read_csv(FROZEN/'all_on/primary_instances.csv');pi=pi[pi.seed>=15]
    stay=pi[pi.algorithm=='milp'].copy();m=ins.set_index(['size','seed'])
    p0=m.loc[list(zip(stay['size'],stay.seed)),'incumbent_w'].to_numpy()
    opt=m.loc[list(zip(stay['size'],stay.seed)),'optimum_w'].to_numpy()
    stay['algorithm']='keep_incumbent'
    for c in METRICS:stay[c]=0.
    for c in ['quality','optimal','feasible','timely','useful','p_quality','p_optimal','ready_s','arrival_s']:stay[c]=np.nan
    stay['network_j']=p0*60;stay['total_j']=p0*60;stay['applied_quality']=((p0-opt)/(opt+1)<=.05).astype(float)
    stay['fallback_loss_j']=60*(p0-opt)
    both=pd.concat([pi,stay]);both.to_csv(out/'primary_instances.csv',index=False)
    primary=compare(both)
    # Preserve the already published paired intervals for unchanged primary rows.
    original=pd.read_csv(FROZEN/'all_on/primary_summary.csv').set_index(['algorithm','shots'])
    for j,r in primary.iterrows():
        key=(r.algorithm,int(r.shots))
        if key in original.index:
            for c in ['ci_low_j','ci_high_j','useful_ci_low','useful_ci_high']:
                primary.loc[j,c]=original.loc[key,c]
    primary.to_csv(out/'primary_summary.csv',index=False)
    horizons=[]
    d=pd.read_csv(FROZEN/'all_on/outcomes.csv');d=d[d.seed>=15]
    for h in [1,10,60,300,3600]:
        e=per_instance(d,replace(DEFAULT,horizon=h));s=stay.copy();s['network_j']=p0*h;s['total_j']=p0*h;s['fallback_loss_j']=h*(p0-opt)
        r=compare(pd.concat([e,s]));r['horizon']=h;horizons.append(r)
    pd.concat(horizons).to_csv(out/'horizon_intervals.csv',index=False)
    # Per-component differences reconcile without mixing direct network and loss decompositions.
    marginal=per_instance(d,replace(DEFAULT,boundary='marginal',marginal_fraction=.001))
    marginal.to_csv(out/'marginal_instances.csv',index=False);compare(marginal).to_csv(out/'marginal_summary.csv',index=False)
    components=['preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j','transition_j','network_j']
    decompositions=[]
    for boundary,p in [('allocated',pi),('marginal',marginal)]:
        base=p[p.algorithm=='milp'].set_index(['size','seed'])
        for alg in ['onehot_structured','onehot_transfer','quantum_uniform_onehot']:
            g=p[(p.algorithm==alg)&(p.shots==64)].copy();keys=list(zip(g['size'],g.seed))
            differences=g[components].to_numpy()-base.loc[keys,components].to_numpy()
            assert np.allclose(differences.sum(axis=1),g.total_j-base.loc[keys,'total_j'].to_numpy(),atol=1e-7)
            for j,c in enumerate(components):
                decompositions.append(dict(boundary=boundary,algorithm=alg,component=c,quantum_j=float(g[c].mean()),classical_j=float(base.loc[keys,c].mean()),difference_j=float(differences[:,j].mean())))
    pd.DataFrame(decompositions).to_csv(out/'decomposition.csv',index=False)
    nominal=compare(per_instance(d,replace(DEFAULT,profile='nominal')));nominal.to_csv(out/'nominal_summary.csv',index=False)
    component=compare(per_instance(d,replace(DEFAULT,qpu_w=18691.2)));component.to_csv(out/'component_summary.csv',index=False)
    # Power-only interpolation is exact: occupied time, gate, and actions do not depend on QPU watts.
    grid=pd.read_csv(FROZEN/'all_on/sensitivity.csv');keys=['algorithm','shots','profile','local_w','reuse','horizon','migration_j','boundary','marginal_fraction']
    lo=grid[grid.qpu_w==10000].set_index(keys);hi=grid[grid.qpu_w==20000].set_index(keys).loc[lo.index]
    cg=lo.reset_index()[keys].copy();cg['qpu_w']=18691.2
    for c in ['excess_j','compute_j','total_j']:cg[c]=lo[c].to_numpy()+.86912*(hi[c].to_numpy()-lo[c].to_numpy())
    cg.to_csv(out/'component_grid.csv',index=False)
    # Simple descriptive diagnostics with complete per-instance records.
    head=ins[['size','seed','incumbent_w','optimum_w']].copy();head['headroom_w']=head.incumbent_w-head.optimum_w;head.to_csv(out/'headroom.csv',index=False)
    sleep_rows=[]
    for family in ['sleep_0','sleep_10','sleep_50']:
        refs=pd.read_csv(FROZEN/family/'instances.csv');refs=refs[refs.seed>=15]
        classical=pd.read_csv(FROZEN/family/'classical.csv');classical=classical[classical.algorithm=='milp'].set_index(['size','seed'])
        for r in refs.itertuples():
            i=E.workload(r.size,r.seed,float(family.split('_')[1]));a=E.decode(i,int(r.incumbent));b=E.decode(i,int(classical.loc[(r.size,r.seed),'solution']))
            sleep_rows.append(dict(workload=family,size=r.size,seed=r.seed,prior_sleep=i.metadata['nodes']-len(set(a)),optimum_sleep=i.metadata['nodes']-len(set(b)),consolidates=len(set(b))<len(set(a))))
    pd.DataFrame(sleep_rows).to_csv(out/'sleep_consolidation.csv',index=False)
    post=d.groupby(['size','seed','algorithm','shots']).postprocessing_s.first().unstack('shots');post['extra_1024_vs64_s']=post[1024]-post[64];post.reset_index().to_csv(out/'postprocessing.csv',index=False)
    milp=pi[pi.algorithm=='milp'].compute_j.mean();shots=pd.read_csv(FROZEN/'all_on/circuits.csv');shots=shots[(shots.seed>=15)&(shots.algorithm=='onehot_structured')]
    details=dict(break_even_us=float(milp/20000*1e6),min_structured_us=float(shots.shot_s.min()*1e6),median_n6_us=float(shots[shots['size']==6].shot_s.median()*1e6),
        headroom_mean_w=float(head.headroom_w.mean()),headroom_max_w=float(head.headroom_w.max()),positive_headroom_w=head[head.headroom_w>1e-8].headroom_w.tolist(),
        component_negative_qaoa_cells=int(((cg.excess_j<0)&cg.algorithm.isin(['qaoa_online','qaoa_transfer','onehot_transfer','onehot_structured','onehot_p2'])).sum()),
        component_negative_quantum_cells=int(((cg.excess_j<0)&(~cg.algorithm.isin(['milp','heuristic','local_search']))&(~cg.algorithm.str.startswith('classical'))).sum()))
    (out/'diagnostics.json').write_text(json.dumps(details,indent=2))


def additions(out, campaign=RESULTS):
    if (campaign/'training_diagnosis.csv').exists():
        d=pd.read_csv(campaign/'training_diagnosis.csv');d.groupby(['workload','size','depth','split'])[['uniform','fitted','transferred']].mean().to_csv(out/'training_summary.csv')
        raw=pd.read_csv(campaign/'aligned_samples.csv');p=raw.groupby(['workload','size','seed','algorithm','shots'],as_index=False).mean(numeric_only=True)
        summaries=[]
        for family,g in p.groupby('workload'):
            old=pd.read_csv(FROZEN/family/'primary_instances.csv');old=old[(old.seed>=15)&old.algorithm.isin(['milp','heuristic'])]
            x=compare(pd.concat([old,g]));x['workload']=family;summaries.append(x)
        pd.concat(summaries).to_csv(out/'aligned_summary.csv',index=False)
    if (campaign/'timing.csv').exists():
        d=pd.read_csv(campaign/'timing.csv');d['excess_j']=0.
        for rep,ix in d.groupby('round').groups.items():
            g=d.loc[ix];base=g[g.algorithm=='milp'].set_index(['size','seed']);d.loc[ix,'excess_j']=g.total_j.to_numpy()-base.loc[list(zip(g['size'],g.seed)),'total_j'].to_numpy()
        d.to_csv(out/'timing_paired.csv',index=False)
        rounds=d.groupby(['round','algorithm','shots'])[['runtime_s','excess_j']].mean().reset_index();rounds.to_csv(out/'timing_rounds.csv',index=False)
        # CIs resample instances after averaging timing rounds, not 600 pseudoreplicates.
        avg=d.groupby(['size','seed','algorithm','shots'],as_index=False).mean(numeric_only=True);compare(avg).to_csv(out/'timing_summary.csv',index=False)
    if (campaign/'larger_quality.csv').exists():
        q=pd.read_csv(campaign/'larger_quality.csv');q.groupby(['size','shots']).mean(numeric_only=True).to_csv(out/'larger_summary.csv')
    if (campaign/'scaling.csv').exists():
        d=pd.read_csv(campaign/'scaling.csv');d.groupby(['nodes','functions']).agg(cases=('seed','size'),exact=('exact','sum'),feasible=('feasible','sum'),median_build_s=('construction_s','median'),median_solve_s=('solve_s','median'),median_total_s=('runtime_s','median'),median_greedy_s=('greedy_s','median'),median_gap=('mip_gap','median'),mean_milp_compute_j=('milp_compute_j','mean'),mean_greedy_compute_j=('greedy_compute_j','mean')).to_csv(out/'scaling_summary.csv')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,default=RESULTS);ap.add_argument('--output',type=Path,default=ROOT/'output/analysis/v121');a=ap.parse_args();a.output=output_directory(a.output)
    static(a.output);additions(a.output,a.campaign);print('Revision analysis complete')
if __name__=='__main__':main()
