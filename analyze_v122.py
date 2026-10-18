#!/usr/bin/env python3
"""Round-three paired analyses; all historical inputs are read only."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from revision_v121 import E,DEFAULT,raw_evaluate,per_instance,paired_interval,ROOT
from analyze_v121 import compare


def keep(pi,raw,scenario=DEFAULT):
    s=pi[pi.algorithm=='milp'].copy();refs=raw.drop_duplicates(['size','seed']).set_index(['size','seed']);keys=list(zip(s['size'],s.seed));p=refs.loc[keys,'incumbent_w'].to_numpy();opt=refs.loc[keys,'optimum_w'].to_numpy()
    s['algorithm']='keep_incumbent';s['shots']=0
    for c in ['compute_j','preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j','transition_j','delay_loss_j','quality_loss_j','ready_s','arrival_s','accepted']:s[c]=0.
    for c in ['quality','optimal','feasible','timely','useful','p_quality','p_optimal']:s[c]=np.nan
    s['network_j']=p*scenario.horizon;s['total_j']=s.network_j;s['fallback_loss_j']=(p-opt)*scenario.horizon;s['applied_quality']=((p-opt)/(opt+1)<=.05).astype(float)
    return s


def stats(g,column):
    lo,hi=paired_interval(g,column);return dict(mean=float(g[column].mean()),ci_low=float(lo),ci_high=float(hi),wins=int((g[column]<-1e-8).sum()),instances=len(g))


def analysis(old,prev,out):
    data=pd.read_csv(old/'all_on/outcomes.csv');data=data[data.seed>=15].copy();primary=pd.read_csv(prev/'primary_instances.csv');summary=pd.read_csv(prev/'primary_summary.csv');q=primary[(primary.algorithm=='onehot_structured')&(primary.shots==64)].set_index(['size','seed']);stay=primary[primary.algorithm=='keep_incumbent'].set_index(['size','seed']);diff=(q.total_j-stay.total_j).rename('q_minus_keep_j').reset_index();diff.to_csv(out/'keep_paired.csv',index=False)
    results={'q_minus_keep':stats(diff,'q_minus_keep_j')}
    small=data[((data.algorithm=='onehot_structured')&(data.shots==64))|(data.algorithm=='milp')]
    def horizon(h):
        sc=replace(DEFAULT,horizon=h);p=per_instance(small,sc);k=keep(p,small,sc);return float(p[p.algorithm=='onehot_structured'].total_j.mean()-k.total_j.mean())
    root=brentq(horizon,1,3600,xtol=1e-9);results['crossing_horizon_s']=root;results['crossing_residual_j']=horizon(root)
    pd.DataFrame([dict(horizon_s=h,q_minus_keep_j=horizon(h)) for h in sorted(set([1,10,30,40,60,300,3600,root,root-.001,root+.001]))]).to_csv(out/'crossing.csv',index=False)
    train=pd.read_csv(prev/'training_diagnosis.csv');aligned=pd.read_csv(prev/'aligned_samples.csv');aligned=aligned[aligned.shots==64];counts=[]
    for (family,n),g in train[(train.depth==1)&(train.split=='test')].groupby(['workload','size']):
        a=aligned[(aligned.workload==family)&(aligned['size']==n)].groupby('seed').p_quality.first();u=g.set_index('seed').uniform
        counts.append(dict(workload=family,size=n,instances=len(g),expectation_below=int((g.transferred<g.uniform-1e-12).sum()),useful_below=int((a<u.loc[a.index]-1e-12).sum()),uniform=u.mean(),expectation=g.transferred.mean(),useful=a.mean()))
    pd.DataFrame(counts).to_csv(out/'below_uniform.csv',index=False)
    ts=pd.read_csv(prev/'training_summary.csv');ts[ts.workload=='all_on'].to_csv(out/'training_main.csv',index=False)
    dispersion=train[(train.workload=='all_on')&(train.split=='train')].groupby(['size','depth'])[['resultant','dispersion']].first();dispersion.to_csv(out/'angle_diagnosis.csv')
    # Controlled replay: only candidate outcomes change. Every primary time and charge,
    # including offline allocation, is shared; this is not a new measured pipeline.
    originals=data[(data.algorithm=='onehot_structured')&(data.shots==64)];profile=originals.drop_duplicates(['size','seed']).set_index(['size','seed']);common=[]
    for r in aligned[aligned.workload=='all_on'].itertuples():
        row=profile.loc[(r.size,r.seed)].copy();i=E.workload(r.size,r.seed,incumbent=int(row.incumbent));ok,p=E.score(i,int(r.solution));row['solution']=int(r.solution);row['algorithm']='useful_mass_common';row['weight']=1/30;row['power_w']=p;row['feasible']=ok;row['quality']=bool(r.quality);row['optimal']=bool(np.isclose(p,row.optimum_w));row['p_quality']=r.p_quality;row['p_optimal']=np.nan;row['moves']=int(np.count_nonzero(E.decode(i,int(r.solution))!=E.decode(i,int(row.incumbent))));row['size']=r.size;row['seed']=r.seed;common.append(row)
    common_raw=pd.DataFrame(common);common_raw.to_csv(out/'common_profile_outcomes.csv',index=False)
    common_pi=per_instance(pd.concat([data[data.algorithm.isin(['milp','heuristic'])],originals,common_raw],ignore_index=True));common_pi.to_csv(out/'common_profile_instances.csv',index=False);cs=compare(common_pi)
    old_primary=summary.set_index(['algorithm','shots'])
    for j,r in cs.iterrows():
        key=(r.algorithm,r.shots)
        if key in old_primary.index:cs.loc[j,['ci_low_j','ci_high_j']]=old_primary.loc[key,['ci_low_j','ci_high_j']].to_numpy()
    cs.to_csv(out/'common_profile_summary.csv',index=False)
    a=common_pi[common_pi.algorithm=='useful_mass_common'].set_index(['size','seed']);b=common_pi[common_pi.algorithm=='onehot_structured'].set_index(['size','seed']);d=(a.total_j-b.total_j).rename('difference_j').reset_index();d.to_csv(out/'objective_paired.csv',index=False);results['common_objective_difference']=stats(d,'difference_j')
    demand=pd.read_csv(prev/'demand_outcomes.csv');dpi=per_instance(demand);dk=keep(dpi,demand);dc=compare(pd.concat([dpi,dk],ignore_index=True));
    prior_demand=pd.read_csv(prev/'demand_summary.csv').set_index(['algorithm','shots'])
    for j,r in dc.iterrows():
        key=(r.algorithm,r.shots)
        if key in prior_demand.index:
            dc.loc[j,['ci_low_j','ci_high_j']]=prior_demand.loc[key,['ci_low_j','ci_high_j']].to_numpy()
    dc.to_csv(out/'demand_summary.csv',index=False);pd.concat([dpi,dk]).to_csv(out/'demand_instances.csv',index=False)
    budgets=[]
    for (alg,n),g in primary[primary.algorithm.isin(['milp','heuristic'])].groupby(['algorithm','size']):
        budgets.append(dict(algorithm=alg,size=n,mean_j=g.compute_j.mean(),min_j=g.compute_j.min(),max_j=g.compute_j.max(),mean_budget_us=g.compute_j.mean()/20000*1e6))
    pd.DataFrame(budgets).to_csv(out/'occupancy_budgets.csv',index=False)
    results['greedy_budget_us']=float(primary[primary.algorithm=='heuristic'].compute_j.mean()/20000*1e6)
    scaling=pd.read_csv(prev/'scaling.csv');scaling['greedy_exact_gap_pct']=np.where(scaling.exact,100*(scaling.greedy_power_w-scaling.power_w)/abs(scaling.power_w),np.nan);scaling['greedy_bound_gap_pct']=100*(scaling.greedy_power_w-scaling.bound_w)/abs(scaling.bound_w);scaling['greedy_budget_us']=scaling.greedy_compute_j/20000*1e6;scaling.to_csv(out/'scaling_quality.csv',index=False)
    sr=[]
    for (n,f),g in scaling.groupby(['nodes','functions']):
        sr.append(dict(nodes=n,functions=f,exact=int(g.exact.sum()),feasible=int(g.feasible.sum()),gap_count=int(g.mip_gap.notna().sum()),median_gap=g.mip_gap.median(),max_gap=g.mip_gap.max(),greedy_exact_count=int(g.greedy_exact_gap_pct.notna().sum()),greedy_exact_median_pct=g.greedy_exact_gap_pct.median(),greedy_exact_max_pct=g.greedy_exact_gap_pct.max(),bound_count=int(g.greedy_bound_gap_pct.notna().sum()),greedy_bound_median_pct=g.greedy_bound_gap_pct.median(),greedy_bound_max_pct=g.greedy_bound_gap_pct.max(),greedy_budget_us=g.greedy_budget_us.mean(),within_max_deadline=int((g.runtime_s<=.8).sum())))
    pd.DataFrame(sr).to_csv(out/'scaling_quality_summary.csv',index=False)
    cases=[('Primary','all_on',DEFAULT),('Marginal 0.001','all_on',replace(DEFAULT,boundary='marginal',marginal_fraction=.001)),('Occupied overheads, u=1','all_on',replace(DEFAULT,repetition_s=.0002,setup_s=.1)),('Occupied overheads, u=0.5','all_on',replace(DEFAULT,repetition_s=.0002,setup_s=.1,utilization=.5)),('Nominal queue','all_on',replace(DEFAULT,profile='nominal')),('Component envelope','all_on',replace(DEFAULT,qpu_w=18691.2)),('10 kW','all_on',replace(DEFAULT,qpu_w=10000))]+[(f'Sleep {w} W',f'sleep_{w}',DEFAULT) for w in [0,10,50]]+[('Demand change','demand',DEFAULT)]
    sensitivities=[];all_summaries=[];sensitivity_instances=[]
    for label,family,sc in cases:
        raw=demand if family=='demand' else pd.read_csv(old/family/'outcomes.csv');raw=raw[(raw.seed>=15)&((raw.shots==64)|(raw.shots==0))];p=per_instance(raw,sc);p=pd.concat([p,keep(p,raw,sc)]);c=compare(p);c['scenario']=label;p['scenario']=label;all_summaries.append(c);sensitivity_instances.append(p)
        q=c[c.algorithm=='onehot_structured'].iloc[0];k=c[c.algorithm=='keep_incumbent'].iloc[0];cl=c[c.algorithm.isin(['milp','heuristic','local_search','classical_uniform_full','classical_uniform_onehot'])].sort_values('excess_j').iloc[0]
        sensitivities.append(dict(scenario=label,useful=100*q.useful,q_excess_j=q.excess_j,q_low=q.ci_low_j,q_high=q.ci_high_j,q_compute_j=q.compute_j,keep_excess_j=k.excess_j,best_classical=cl.algorithm,best_classical_excess_j=cl.excess_j,timing_campaign='Primary + Demand' if family=='demand' else 'Primary'))
    pd.DataFrame(sensitivities).to_csv(out/'sensitivity_summary.csv',index=False);pd.concat(all_summaries).to_csv(out/'sensitivity_all.csv',index=False);pd.concat(sensitivity_instances).to_csv(out/'sensitivity_instances.csv',index=False)
    z=data.copy();z.loc[z.algorithm=='onehot_structured','compile_s']=0.;zs=[]
    for boundary in ['allocated','marginal']:
        zp=per_instance(z,replace(DEFAULT,boundary=boundary,marginal_fraction=.001));zc=compare(zp);zc['boundary']=boundary;zs.append(zc)
    pd.concat(zs).to_csv(out/'zero_transpilation_summary.csv',index=False)
    (out/'diagnostics.json').write_text(json.dumps(results,indent=2))
    if (out/'template_timing.csv').exists() and (out/'template_equivalence.csv').exists():
        times=pd.read_csv(out/'template_timing.csv').groupby(['size','seed']).mean(numeric_only=True);offline=pd.read_csv(out/'template_offline.csv').groupby('size').total_s.mean();ledgers=[]
        for mode in ['template','numeric']:
            raw=data.copy();raw=raw[((raw.algorithm=='onehot_structured')&(raw.shots==64))|raw.algorithm.isin(['milp','heuristic'])].copy()
            for (n,seed),ix in raw[raw.algorithm=='onehot_structured'].groupby(['size','seed']).groups.items():
                t=times.loc[(n,seed)];raw.loc[ix,'dense_preparation_s']=t.preparation_s;raw.loc[ix,'circuit_build_s']=t.binding_s if mode=='template' else t.numeric_build_s;raw.loc[ix,'compile_s']=0 if mode=='template' else t.numeric_compile_s;raw.loc[ix,'shot_s']=t[f'{mode}_shot_s']
                if mode=='template':raw.loc[ix,'offline_s']+=offline.loc[n]
            for boundary in ['allocated','marginal']:
                sc=replace(DEFAULT,boundary=boundary,marginal_fraction=.001);p=per_instance(raw,sc);p['mode']=mode;p['boundary']=boundary;ledgers.append(p)
        lp=pd.concat(ledgers);lp.to_csv(out/'template_instances.csv',index=False)
        ss=[]
        for (mode,boundary),p in lp.groupby(['mode','boundary']):
            s=compare(p);s['mode']=mode;s['boundary']=boundary;ss.append(s)
        pd.concat(ss).to_csv(out/'template_summary.csv',index=False)
    manifest=dict(bootstrap_resamples=20000,bootstrap_seed=20261017,unit='size-stratified problem instance; repetitions averaged',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'revision_v121.py',ROOT/'analyze_v12.py']})
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(results,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=ROOT/'results/v1_2');ap.add_argument('--previous',type=Path,default=ROOT/'results/v1_2_1');ap.add_argument('--output',type=Path,default=ROOT/'results/v1_2_2');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);analysis(a.input,a.previous,a.output)
if __name__=='__main__':main()
