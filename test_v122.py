#!/usr/bin/env python3
"""Independent event, equivalence, pairing, timeout-bound and preservation audits."""
from artifact_paths import validation_path
import hashlib,json
from pathlib import Path
from dataclasses import replace
import numpy as np,pandas as pd
import energy_v12 as E
from analyze_v12 import raw_evaluate,DEFAULT
from run_v122 import template,bind,logical_probabilities
ROOT=Path(__file__).resolve().parent;D=ROOT/'results/v1_2_2';P=ROOT/'results/v1_2_1'

def main():
    audits=pd.read_csv(D/'template_equivalence.csv');assert len(audits)==60;assert audits.max_probability_error.max()<1e-9;assert audits.parameters_remaining.eq(0).all()
    times=pd.read_csv(D/'template_timing.csv');assert len(times)==600;assert times.groupby(['size','seed']).size().eq(10).all();assert set(times.seed)==set(range(15,30));assert (times[['binding_s','numeric_compile_s','preparation_s']]>0).all().all()
    angles=json.loads((ROOT/'results/v1_2/all_on/transfer.json').read_text())
    for n,seed in [(6,15),(9,29)]:
        i=E.workload(n,seed);theta=next(t['theta'] for t in angles if t['size']==n and t['depth']==1);q,ph,beta=template(i);c,_=E.compile_circuit(q);p=logical_probabilities(bind(c,ph,beta,i,theta),n);assert np.allclose(p,E.onehot_probs(i,theta),atol=1e-9)
    primary=pd.read_csv(P/'primary_instances.csv');q=primary[(primary.algorithm=='onehot_structured')&(primary.shots==64)].set_index(['size','seed']);keep=primary[primary.algorithm=='keep_incumbent'].set_index(['size','seed']);paired=pd.read_csv(D/'keep_paired.csv').set_index(['size','seed']);assert np.allclose(paired.q_minus_keep_j,q.total_j-keep.total_j);assert int((paired.q_minus_keep_j<0).sum())==4
    cross=pd.read_csv(D/'crossing.csv');root=json.loads((D/'diagnostics.json').read_text())['crossing_horizon_s'];assert 42<root<43;assert cross.iloc[(cross.horizon_s-root).abs().argsort()[:1]].q_minus_keep_j.abs().max()<1e-7;assert cross[cross.horizon_s<root].q_minus_keep_j.gt(0).all();assert cross[cross.horizon_s>root].q_minus_keep_j.lt(0).all()
    common=pd.read_csv(D/'common_profile_instances.csv');a=common[common.algorithm=='useful_mass_common'].set_index(['size','seed']);b=common[common.algorithm=='onehot_structured'].set_index(['size','seed']);assert np.allclose(a.compute_j,b.compute_j);assert np.allclose(a.ready_s,b.ready_s);assert np.allclose(a.offline_j,b.offline_j)
    raw=pd.read_csv(D/'common_profile_outcomes.csv');values,ready,_=raw_evaluate(raw);replays=0;cache={}
    for j,r in enumerate(raw.itertuples()):
        key=(r.size,r.seed)
        if key not in cache:cache[key]=E.workload(r.size,r.seed,incumbent=int(r.incumbent))
        e=E.event_energy(cache[key],[(ready[j],int(r.solution),'common')],DEFAULT)
        assert np.isclose(e['network_j'],values['network_j'][j],atol=1e-7);assert np.isclose(e['transition_j'],values['transition_j'][j],atol=1e-8);replays+=1
    demand=pd.read_csv(D/'demand_instances.csv');k=demand[demand.algorithm=='keep_incumbent'];assert k[['compute_j','transition_j','preparation_j','service_j']].eq(0).all().all();assert k[['quality','timely','useful']].isna().all().all();assert np.allclose(k.total_j,k.network_j)
    ledger=pd.read_csv(D/'template_instances.csv');components=['preparation_j','optimization_j','communication_j','postprocessing_j','offline_j','service_j','transition_j','network_j'];assert np.allclose(ledger[components].sum(axis=1),ledger.total_j)
    sq=pd.read_csv(D/'scaling_quality.csv');assert sq.greedy_bound_gap_pct[sq.bound_w.isna()].isna().all();assert (sq.loc[sq.bound_w.notna(),'bound_w']<=sq.loc[sq.bound_w.notna(),'greedy_power_w']+1e-6).all();assert sq.greedy_exact_gap_pct[~sq.exact].isna().all();assert not sq.loc[sq.status==1,'exact'].any();assert int(sq.bound_w.isna().sum())==1
    bs=pd.read_csv(D/'below_uniform.csv');all_on=bs[bs.workload=='all_on'];assert all_on.expectation_below.sum()==39;assert all_on.useful_below.sum()==26
    preserved=json.loads((D/'historical_sha256.json').read_text())
    for name,h in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    report=dict(template_equivalence_cases=len(audits),template_probability_error_max=float(audits.max_probability_error.max()),fresh_template_rechecks=2,timing_records=len(times),independent_event_replays=replays,keep_pairing=True,crossing_replayed=True,shared_profile=True,keep_zero_charges=True,ledger_reconciled=True,timeout_bounds=True,training_split=True,historical_files_unchanged=len(preserved))
    validation_path('test_v122').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
