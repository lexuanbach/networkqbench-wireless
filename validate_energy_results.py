#!/usr/bin/env python3
"""Independent audit of stored sustainability results and phase accounting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from energy_model import energy_placement, applied_energy, server_power, LOAD, WATTS

def main():
    path=Path(__file__).resolve().parent/"results"/"sustainability"
    m=json.loads((path/"manifest.json").read_text())
    assert (m["instances"],m["test_instances"],m["matched_circuits"],m["sampling_rows"])==(120,60,360,43200)
    ins=pd.read_csv(path/"instances.csv");cl=pd.read_csv(path/"classical.csv")
    raw=pd.read_csv(path/"samples.csv");cir=pd.read_csv(path/"circuits.csv")
    assert not cir.duplicated(["size","seed","algorithm"]).any()
    assert len(cir)==360 and len(raw)==43200
    np.testing.assert_allclose(server_power(LOAD),WATTS)
    instances={}
    for r in ins.itertuples():
        i=energy_placement(r.size,r.seed);instances[r.size,r.seed]=i
        np.testing.assert_allclose(i.metadata["power_w"][i.feasible].min(),r.optimum_w,atol=1e-8)
    for r in cl[cl.algorithm=="milp"].itertuples():
        assert instances[r.size,r.seed].feasible[r.solution]
        np.testing.assert_allclose(r.power_w,r.optimum_w,atol=1e-8)
    for r in raw.itertuples():
        i=instances[r.size,r.seed]
        assert bool(i.feasible[r.solution])==r.feasible
        np.testing.assert_allclose(i.metadata["power_w"][r.solution],r.power_w,atol=1e-8)
    transfer=json.loads((path/"transfer.json").read_text())
    assert all(x["training_seeds"]==list(range(15)) for x in transfer)
    fit=pd.read_csv(path/"fits.csv")
    for t in transfer:
        g=fit[(fit["size"]==t["size"])&(fit.mixer==t["mixer"])&(fit.seed<15)]
        for key,period in [("gamma",2*np.pi),("beta",np.pi)]:
            expected=np.angle(np.exp(2j*np.pi*g[key].to_numpy()/period).mean())%(2*np.pi)*period/(2*np.pi)
            np.testing.assert_allclose(expected,t[key],atol=1e-12)
    primary=pd.read_csv(path/"primary_decisions.csv")
    components=[c for c in primary if c.startswith("compute_") and c!="compute_j"]
    np.testing.assert_allclose(primary[components].sum(axis=1),primary.compute_j,rtol=1e-12)
    np.testing.assert_allclose(primary.compute_j+primary.network_j+primary.transition_j,primary.total_j,rtol=1e-12)
    expected=[]
    index=primary.set_index(["size","seed","algorithm","shots"])
    for r in raw.itertuples():
        p=index.loc[(r.size,r.seed,r.algorithm,r.shots)]
        e=applied_energy(instances[r.size,r.seed],r.solution,float(p.delay_s),60,10)
        expected.append((r.size,r.seed,r.algorithm,r.shots,e["network_j"],e["migration_j"]))
    replay=pd.DataFrame(expected,columns=["size","seed","algorithm","shots","network","migration"]).groupby(["size","seed","algorithm","shots"]).mean()
    aligned=index.loc[replay.index]
    np.testing.assert_allclose(replay.network,aligned.network_j,rtol=1e-12)
    np.testing.assert_allclose(replay.migration,aligned.transition_j,rtol=1e-12,atol=1e-10)
    scenario=pd.read_csv(path/"sensitivity.csv")
    assert len(scenario)==4320*15
    assert not (scenario[scenario.algorithm.str.contains("qaoa|onehot")].saving_vs_milp_j>1e-8).any()
    pol=pd.read_csv(path/"policies.csv");assert set(pol.seed)==set(range(15,30))
    assert not pol[pol.policy=="energy_aware"].invoked_quantum.any()
    decisions=pd.read_csv(path/"admission_decisions.csv")
    px=pol.set_index(["size","seed","policy"])
    for r in decisions.itertuples():
        i=instances[r.size,r.seed];p=px.loc[(r.size,r.seed,r.policy)]
        expected=r.computed_action if p.on_time and i.feasible[r.computed_action] else i.metadata["incumbent"]
        assert r.applied_action==expected
    report={"instances_verified":120,"milp_matches":120,"sampling_rows_verified":len(raw),
            "independent_applied_energy_replays":len(raw),"scenario_summaries":len(scenario),
            "policy_decisions_verified":len(decisions),"training_membership":"0--14 only",
            "status":"passed"}
    (path/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
