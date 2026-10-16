#!/usr/bin/env python3
"""Regenerate scenario analyses, policy evaluation and manuscript figures."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from energy_model import PROFILES, minimum_shots

ROOT=Path(__file__).resolve().parent
DEFAULT=dict(profile="optimistic",local_w=50.,qpu_w=20000.,reuse=100,horizon=60.,migration=10.,boundary="allocated",marginal_fraction=.01)
LABELS={"milp":"HiGHS MILP","local_search":"Local search","heuristic":"Greedy placement",
        "qaoa_online":"Online QAOA","qaoa_transfer":"Transferred QAOA","onehot_transfer":"One-hot transfer"}


def prepare(path):
    raw=pd.read_csv(path/"samples.csv")
    moved=np.zeros(len(raw))
    for size,(f,n) in {6:(2,3),8:(2,4),9:(3,3),12:(3,4)}.items():
        ix=raw["size"].eq(size).to_numpy();s=raw.loc[ix,"solution"].to_numpy(dtype=np.int64)
        prev=raw.loc[ix,"incumbent"].to_numpy(dtype=np.int64)
        moved[ix]=sum(((s>>(k*n))&((1<<n)-1))!=((prev>>(k*n))&((1<<n)-1)) for k in range(f))
    raw["feasible_power"]=raw.power_w*raw.feasible
    raw["feasible_moves"]=moved*raw.feasible
    g=raw.groupby(["size","seed","algorithm","shots"],as_index=False).agg(
        split=("split","first"),optimum_w=("optimum_w","first"),incumbent_w=("incumbent_w","first"),
        deadline_s=("deadline_s","first"),dense_preparation_s=("dense_preparation_s","first"),
        feasible_rate=("feasible","mean"),quality_rate=("quality","mean"),optimal_rate=("optimal","mean"),
        feasible_power=("feasible_power","mean"),feasible_moves=("feasible_moves","mean"),
        p_quality=("p_quality","first"),p_optimal=("p_optimal","first"),evals=("evals","first"),
        postprocessing_s=("postprocessing_s","median"),offline_s=("offline_s","first"))
    g=g.merge(pd.read_csv(path/"circuits.csv"),on=["size","seed","algorithm"],validate="many_to_one")
    g["local_runtime_s"]=0.
    cl=pd.read_csv(path/"classical.csv")
    for name,val in {"shots":0,"feasible_rate":1.,"evals":0,"dense_preparation_s":0.,"postprocessing_s":0.,
                     "offline_s":0.,"circuit_build_s":0.,"compile_s":0.,"shot_s":0.,"cx":0,"depth":0}.items():cl[name]=val
    cl["local_runtime_s"]=cl.runtime_s
    cl["quality_rate"]=((cl.power_w-cl.optimum_w)/(abs(cl.optimum_w)+1)<=.05).astype(float)
    cl["optimal_rate"]=(cl.power_w-cl.optimum_w<=1e-8).astype(float)
    cl["p_quality"]=cl.quality_rate;cl["p_optimal"]=cl.optimal_rate;cl["feasible_power"]=cl.power_w
    cl["feasible_moves"]=0.
    for size,(f,n) in {6:(2,3),8:(2,4),9:(3,3),12:(3,4)}.items():
        ix=cl["size"].eq(size).to_numpy();s=cl.loc[ix,"solution"].to_numpy(dtype=np.int64);prev=cl.loc[ix,"incumbent"].to_numpy(dtype=np.int64)
        cl.loc[ix,"feasible_moves"]=sum(((s>>(k*n))&((1<<n)-1))!=((prev>>(k*n))&((1<<n)-1)) for k in range(f))
    return pd.concat([g,cl[g.columns.intersection(cl.columns)]],ignore_index=True)


def evaluate(data,profile="optimistic",local_w=50.,qpu_w=20000.,reuse=100,horizon=60.,migration=10.,boundary="allocated",marginal_fraction=.01):
    q=data.algorithm.str.contains("qaoa|onehot").to_numpy()
    queue,submit,roundtrip=PROFILES[profile]
    z=data.copy()
    prep=z.dense_preparation_s+z.circuit_build_s+z.compile_s
    opt_host=np.maximum(z.evals-1,0)*.0001
    comm=z.evals*(submit+roundtrip)
    service=z.evals*z.shots*z.shot_s
    z["delay_s"]=np.where(q,prep+opt_host+comm+service+z.postprocessing_s+queue,z.local_runtime_s)
    z["compute_preparation_j"]=np.where(q,local_w*prep,0)
    z["compute_optimization_j"]=np.where(q,local_w*opt_host,local_w*z.local_runtime_s)
    z["compute_communication_j"]=local_w*comm
    z["compute_postprocessing_j"]=local_w*z.postprocessing_s
    z["compute_offline_j"]=local_w*z.offline_s/reuse
    z["compute_service_j"]=qpu_w*service*(marginal_fraction if boundary=="marginal" else 1.)
    z["compute_j"]=z[[c for c in z if c.startswith("compute_")]].sum(axis=1)
    ontime=(z.delay_s<=z.deadline_s)&(z.delay_s<horizon)
    z["on_time"]=ontime
    z["accepted_rate"]=ontime*z.feasible_rate
    z["useful_rate"]=ontime*z.quality_rate
    arrival=np.minimum(z.delay_s,horizon)
    average_applied=z.feasible_power+(1-z.feasible_rate)*z.incumbent_w
    z["network_j"]=np.where(ontime,z.incumbent_w*arrival+average_applied*(horizon-arrival),z.incumbent_w*horizon)
    z["transition_j"]=np.where(ontime,z.feasible_moves*migration,0.)
    z["total_j"]=z.compute_j+z.network_j+z.transition_j
    z["no_control_j"]=z.incumbent_w*horizon
    z["saving_vs_no_control_j"]=z.no_control_j-z.total_j
    baseline=z[z.algorithm=="milp"][["size","seed","total_j"]].rename(columns={"total_j":"milp_total_j"})
    z=z.merge(baseline,on=["size","seed"],validate="many_to_one")
    z["saving_vs_milp_j"]=z.milp_total_j-z.total_j
    z["energy_evidence"]="modeled_energy"
    z["network_evidence"]="modeled_network"
    z["timing_evidence"]=np.where(z.algorithm.str.contains("qaoa|onehot"),"modeled_qpu","measured_local")
    for k,v in dict(profile=profile,local_w=local_w,qpu_w=qpu_w,reuse=reuse,horizon=horizon,migration=migration,boundary=boundary,marginal_fraction=marginal_fraction).items():z[k]=v
    return z


def summarize(z):
    s=z.groupby(["algorithm","shots"],as_index=False).agg(instances=("seed","size"),
       feasibility=("feasible_rate","mean"),quality=("quality_rate","mean"),optimal=("optimal_rate","mean"),
       on_time=("on_time","mean"),useful=("useful_rate","mean"),compute_j=("compute_j","mean"),
       network_j=("network_j","mean"),transition_j=("transition_j","mean"),total_j=("total_j","mean"),
       saving_vs_milp_j=("saving_vs_milp_j","mean"),saving_vs_no_control_j=("saving_vs_no_control_j","mean"))
    s["energy_per_useful_j"]=np.divide(s.compute_j,s.useful,out=np.full(len(s),np.inf),where=s.useful>0)
    for k in DEFAULT:s[k]=z[k].iloc[0]
    s["energy_evidence"]="modeled_energy"
    s["network_evidence"]="modeled_network"
    return s


def paired_bootstrap(z):
    rng=np.random.default_rng(20261015);rows=[]
    for (alg,shots),g in z.groupby(["algorithm","shots"]):
        samples=[]
        for size,sg in g.groupby("size"):
            x=sg.saving_vs_milp_j.to_numpy();samples.append(x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1))
        means=np.mean(samples,axis=0)
        rows.append({"algorithm":alg,"shots":int(shots),"mean_saving_j":float(g.saving_vs_milp_j.mean()),
                     "ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975)),
                     "unit":"held-out instance; equal weight to each size; 20000 paired resamples"})
    return pd.DataFrame(rows)


def policies(data,path,primary):
    raw=pd.read_csv(path/"samples.csv");cl=pd.read_csv(path/"classical.csv")
    fit=pd.read_csv(path/"fits.csv");cir=pd.read_csv(path/"circuits.csv");ins=pd.read_csv(path/"instances.csv")
    source_seconds={}
    for size in sorted(ins["size"].unique()):
        source_seconds[size]=float(fit[(fit["size"]==size)&(fit.seed<15)].simulation_s.sum()+
            cir[(cir["size"]==size)&(cir.seed<15)][["compile_s","circuit_build_s"]].sum().sum()+
            cl[(cl["size"]==size)&(cl.seed<15)].runtime_s.sum()+
            ins[(ins["size"]==size)&(ins.seed<15)].dense_preparation_s.sum())
    timing_path=path/"policy_timing.json"
    timing=json.loads(timing_path.read_text()) if timing_path.exists() else {}
    start=time.perf_counter()
    train=primary[primary.seed<15]
    model=train.groupby(["size","algorithm","shots"],as_index=False).agg(
        predicted_quality=("quality_rate","mean"),predicted_delay=("delay_s",lambda x:float(x.quantile(.9))),
        predicted_total=("total_j","mean"))
    calibration_s=timing.get("calibration_s",time.perf_counter()-start)
    timing["calibration_s"]=calibration_s
    model.to_csv(path/"admission_calibration.csv",index=False)
    rows=[];decisions=[]
    for size in sorted(data["size"].unique()):
        candidates=model[model["size"]==size]
        for seed in sorted(data[(data["size"]==size)&(data.seed>=15)].seed.unique()):
            group=primary[(primary["size"]==size)&(primary.seed==seed)]
            deadline=float(group.deadline_s.iloc[0])
            for policy in ["always_classical","always_quantum","deadline_only","energy_aware"]:
                start=time.perf_counter(); reason="fixed comparator";chosen=("milp",0)
                if policy=="always_quantum":chosen=("qaoa_transfer",1024)
                elif policy in ["deadline_only","energy_aware"]:
                    eligible=candidates[(candidates.predicted_quality>=.9)&(candidates.predicted_delay<=deadline)]
                    if policy=="deadline_only":
                        quantum=eligible[eligible.algorithm.str.contains("qaoa|onehot")]
                        if len(quantum):
                            best=quantum.sort_values(["predicted_delay","algorithm","shots"]).iloc[0];chosen=(best.algorithm,int(best.shots));reason="training quality and delay pass"
                        else:reason="no quantum configuration meets training quality and delay"
                    else:
                        # Strong classical reference remains a candidate; no held-out quality is read.
                        eligible=eligible[(eligible.algorithm=="milp")|eligible.algorithm.str.contains("qaoa|onehot")]
                        if len(eligible):
                            best=eligible.sort_values(["predicted_total","algorithm","shots"]).iloc[0];chosen=(best.algorithm,int(best.shots))
                        reason="training total-energy ranking" if chosen[0]!="milp" else "no predicted quantum energy benefit"
                timing_key=f"{int(size)}:{int(seed)}:{policy}"
                lookup_s=timing.get(timing_key,time.perf_counter()-start)
                timing[timing_key]=lookup_s
                selected=group[(group.algorithm==chosen[0])&(group.shots==chosen[1])].iloc[0]
                pred=candidates[(candidates.algorithm==chosen[0])&(candidates.shots==chosen[1])].iloc[0]
                # Policies that require calibration pay for the complete offline local
                # training/compilation dataset, replacing the selected solver's training.
                training_j=(source_seconds[size]+calibration_s)*50/100 if policy in ["deadline_only","energy_aware"] else float(selected.compute_offline_j)
                extra=lookup_s*50+training_j-float(selected.compute_offline_j)
                delay=float(selected.delay_s)+lookup_s
                on_time=delay<=deadline and delay<DEFAULT["horizon"]
                average_power=float(selected.feasible_power+(1-selected.feasible_rate)*selected.incumbent_w)
                network=(selected.incumbent_w*delay+average_power*(DEFAULT["horizon"]-delay)) if on_time else selected.incumbent_w*DEFAULT["horizon"]
                transition=float(selected.feasible_moves)*DEFAULT["migration"] if on_time else 0.
                total=float(selected.compute_j)+extra+network+transition
                rows.append({"size":int(size),"seed":int(seed),"policy":policy,"algorithm":chosen[0],"shots":chosen[1],
                             "invoked_quantum":chosen[0]!="milp","lookup_s":lookup_s,"training_j":training_j,
                             "compute_j":float(selected.compute_j)+extra,"total_j":total,"delay_s":delay,
                             "useful_rate":float(selected.quality_rate)*on_time,"on_time":bool(on_time),
                             "saving_vs_milp_j":float(selected.milp_total_j)-total,"reason":reason})
                trials=raw[(raw["size"]==size)&(raw.seed==seed)&(raw.algorithm==chosen[0])&(raw.shots==chosen[1])]
                if chosen[0]=="milp":trials=cl[(cl["size"]==size)&(cl.seed==seed)&(cl.algorithm=="milp")].assign(repeat=0,feasible=True)
                for t in trials.itertuples():
                    applied=int(t.solution) if on_time and t.feasible else int(t.incumbent)
                    decisions.append({"size":int(size),"seed":int(seed),"repeat":int(t.repeat),"policy":policy,
                                      "selected_solver":chosen[0],"shots":chosen[1],"rejection_reason":reason,
                                      "predicted_quality_probability":float(pred.predicted_quality),
                                      "predicted_delay_s":float(pred.predicted_delay),"predicted_total_energy_j":float(pred.predicted_total),
                                      "computed_action":int(t.solution),"applied_action":applied})
    pd.DataFrame(decisions).to_csv(path/"admission_decisions.csv",index=False)
    result=pd.DataFrame(rows);result.to_csv(path/"policies.csv",index=False)
    (path/"admission_manifest.json").write_text(json.dumps({"training_seeds":list(range(15)),"test_seeds":list(range(15,30)),
        "calibration_s":calibration_s,"offline_seconds_by_size":{str(k):v for k,v in source_seconds.items()},"quality_target":.9,
        "delay_statistic":"training 90th percentile","energy_statistic":"training mean",
        "warning":"empirical policy; predictions have no per-instance guarantee",
        "default_scenario":DEFAULT},indent=2)+"\n")
    timing_path.write_text(json.dumps(timing,indent=2)+"\n")
    return result


def legacy_reanalysis(path):
    old=ROOT/"results";r=pd.read_csv(old/"raw_results.csv")
    r["candidate_shots"]=r.shots.fillna(0)
    r["modeled_execution_count"]=np.where(r.algorithm=="qaoa",r.evals*r.candidate_shots,0)
    # Complete matched circuits are unavailable for historical rows: keep energy unknown.
    r["local_energy_10w_j"]=np.where(r.algorithm!="qaoa",r.runtime_s*10,np.nan)
    r["local_energy_50w_j"]=np.where(r.algorithm!="qaoa",r.runtime_s*50,np.nan)
    r["local_energy_100w_j"]=np.where(r.algorithm!="qaoa",r.runtime_s*100,np.nan)
    r["qpu_energy_j"]=np.nan
    r["energy_status"]=np.where(r.algorithm=="qaoa","unknown: no instance-matched complete circuit","modeled local power times measured runtime")
    r.to_csv(path/"legacy_reanalysis.csv",index=False)


def extra_diagnostics(path,data,primary):
    shots=[]
    unique=data[data.algorithm.str.contains("qaoa|onehot")].drop_duplicates(["size","seed","algorithm"])
    for r in unique.itertuples():
        for target in [.90,.95]:
            for tolerance,p in [(0.,r.p_optimal),(.05,r.p_quality)]:
                need=minimum_shots(p,target)
                shots.append({"size":r.size,"seed":r.seed,"algorithm":r.algorithm,"epsilon":tolerance,
                              "alpha":target,"per_shot_probability":p,"minimum_shots":need,
                              "evidence":"exact_simulated; independent fixed-distribution draws"})
    pd.DataFrame(shots).to_csv(path/"shot_requirements.csv",index=False)
    raw=pd.read_csv(path/"samples.csv")
    rates=raw.groupby(["size","seed","algorithm","shots"],as_index=False).agg(n=("quality","size"),hits=("quality","sum"))
    z=1.959963984540054;rate=rates.hits/rates.n;denom=1+z*z/rates.n
    center=(rate+z*z/(2*rates.n))/denom
    half=z*np.sqrt(rate*(1-rate)/rates.n+z*z/(4*rates.n**2))/denom
    rates["quality_rate"]=rate;rates["wilson_low"]=center-half;rates["wilson_high"]=center+half
    rates["unit"]="independent repeated shot batches within one fixed instance/distribution"
    rates.to_csv(path/"sampling_intervals.csv",index=False)
    reuse=[]
    for r in primary[(primary.seed>=15)&primary.algorithm.str.contains("qaoa|onehot")].itertuples():
        for comparator in ["milp","heuristic"]:
            c=primary[(primary["size"]==r.size)&(primary.seed==r.seed)&(primary.algorithm==comparator)].iloc[0]
            room=float(c.total_j)-(r.total_j-r.compute_offline_j)
            offline=r.compute_offline_j*DEFAULT["reuse"]
            need=max(1,math.floor(offline/room)+1) if room>0 and r.on_time else np.nan
            reuse.append({"size":r.size,"seed":r.seed,"algorithm":r.algorithm,"shots":r.shots,"comparator":comparator,
                          "minimum_reuse_for_strict_total_energy_gain":need,"offline_total_j":offline,"headroom_before_training_j":room,
                          "quality_rate":r.quality_rate,"reason":"finite conditional threshold" if np.isfinite(need) else "online cost or deadline prevents repayment"})
    pd.DataFrame(reuse).to_csv(path/"reuse_thresholds.csv",index=False)


def figures(path,primary,sensitivity,policy):
    dest=ROOT/"output"/"sustainability";dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(10,3.6));ax.set(xlim=(0,10),ylim=(0,3.6));ax.axis("off")
    boxes=[(.1,1.9,1.8,1.1,"Network state\nIncumbent + deadline"),(2.2,1.9,2.,1.1,"Admission\nQuality / time / energy"),
           (4.6,2.5,2.1,.8,"Classical solver"),(4.6,1.3,2.1,.8,"Quantum service\nPrepare / sample / return"),
           (7.2,1.9,2.5,1.1,"Apply timely feasible action\nOtherwise retain incumbent")]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.05",facecolor="#eaf2f8",edgecolor="#24577a"));ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9)
    for a,b in [((1.9,2.45),(2.15,2.45)),((4.25,2.5),(4.55,2.85)),((4.25,2.35),(4.55,1.75)),((6.75,2.85),(7.15,2.6)),((6.75,1.75),(7.15,2.3))]:ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->"})
    ax.text(5,.65,"Computation ledger: preparation + host work + active QPU service + offline / M",ha="center")
    ax.text(5,.2,"Application ledger: computation + migration + server power integrated over action lifetime",ha="center")
    fig.savefig(dest/"accounting.pdf",bbox_inches="tight");plt.close(fig)
    s=summarize(primary[primary.seed>=15])
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for alg in LABELS:
        g=s[s.algorithm==alg].sort_values("shots")
        axs[0].plot(g.useful,g.compute_j,"o-",label=LABELS[alg])
        if alg == "onehot_transfer":
            for r in g.itertuples():axs[0].annotate(str(r.shots),(r.useful,r.compute_j),fontsize=7,xytext=(3,3),textcoords="offset points")
    axs[0].set(xlabel="Timely, feasible, 5%-quality probability",ylabel="Mean computation energy (J)",yscale="log",xlim=(-.05,1.05))
    axs[0].legend(fontsize=7,loc="best")
    cs=pd.read_csv(path/"circuits.csv");cs=cs[cs.seed>=15]
    for alg in ["qaoa_online","qaoa_transfer","onehot_transfer"]:
        gg=cs[cs.algorithm==alg].groupby("size").shot_s.median()
        axs[1].plot(gg.index,gg*1e6,"o-",label=LABELS[alg])
    axs[1].set(xlabel="Binary variables",ylabel="Median scheduled shot duration (microseconds)",yscale="log");axs[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(dest/"energy_quality.pdf");plt.close(fig)
    # Transparent diagnostic frontier: a fixed transferred n=6 circuit, without averaging p across instances.
    row=primary[(primary["size"]==6)&(primary.seed==15)&(primary.algorithm=="onehot_transfer")&(primary.shots==64)].iloc[0]
    ds=np.linspace(0,1,151);bs=np.geomspace(.1,1000,151)
    time_fixed=row.delay_s-row.shots*row.shot_s
    energy_fixed=row.compute_j-row.shots*row.shot_s*20000
    need=minimum_shots(row.p_quality)
    cap=np.minimum(np.maximum(0,np.floor((ds[None,:]-time_fixed)/row.shot_s)),np.maximum(0,np.floor((bs[:,None]-energy_fixed)/(row.shot_s*20000))))
    fig,ax=plt.subplots(figsize=(6,3.7));ax.pcolormesh(ds,bs,(cap>=need).astype(float),cmap="Blues",vmin=0,vmax=1,shading="auto")
    ax.set(yscale="log",xlabel="Decision deadline (s)",ylabel="Computation energy budget (J)",title=f"One-hot transfer, n=6, seed=15; minimum {int(need)} shots")
    fig.tight_layout();fig.savefig(dest/"joint_frontier.pdf");plt.close(fig)
    subset=sensitivity[(sensitivity.profile=="optimistic")&(sensitivity.local_w==50)&(sensitivity.qpu_w==20000)&(sensitivity.reuse==100)&(sensitivity.migration==10)&(sensitivity.boundary=="allocated")]
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for alg,shots in [("milp",0),("heuristic",0),("qaoa_transfer",64),("onehot_transfer",64),("onehot_transfer",1024)]:
        g=subset[(subset.algorithm==alg)&(subset.shots==shots)].sort_values("horizon")
        label=LABELS[alg]+(f", S={shots}" if shots else "")
        axs[0].plot(g.horizon,g.saving_vs_no_control_j,"o-",label=label)
        axs[1].plot(g.horizon,g.saving_vs_milp_j,"o-",label=label)
    for ax in axs:ax.set(xscale="log",xlabel="Constant-demand action lifetime (s)");ax.axhline(0,color="gray",lw=.7)
    axs[0].set_ylabel("Energy saving vs retained incumbent (J)");axs[1].set_ylabel("Energy saving vs MILP (J)")
    axs[0].legend(fontsize=6);fig.tight_layout();fig.savefig(dest/"net_energy.pdf");plt.close(fig)
    p=policy.groupby("policy").agg(extra=("saving_vs_milp_j",lambda x:-x.mean()),useful=("useful_rate","mean"),invoke=("invoked_quantum","mean"))
    order=["always_classical","always_quantum","deadline_only","energy_aware"];p=p.loc[order]
    fig,axs=plt.subplots(1,2,figsize=(10,3.7));x=np.arange(len(p));names=["Classical","Quantum","Deadline","Energy"]
    axs[0].bar(x,p.extra,color="#3878a8");axs[0].set(yscale="symlog",ylabel="Extra total energy vs MILP (J)",xticks=x,xticklabels=names)
    axs[1].bar(x-.18,p.useful,.36,label="Useful decision");axs[1].bar(x+.18,p.invoke,.36,label="Quantum invocation")
    axs[1].set(ylim=(0,1.1),ylabel="Fraction",xticks=x,xticklabels=names);axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/"admission.pdf");plt.close(fig)
    return dest


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--results",type=Path,default=ROOT/"results"/"sustainability");ap.add_argument("--reuse-sensitivity",action="store_true",help="Reuse stored scenario sweep when only plots/policies changed");args=ap.parse_args();path=args.results
    data=prepare(path);data.to_csv(path/"instance_outcomes.csv",index=False)
    primary=evaluate(data,**DEFAULT);primary.to_csv(path/"primary_decisions.csv",index=False)
    summary=summarize(primary[primary.seed>=15]);summary.to_csv(path/"primary_summary.csv",index=False)
    ci=paired_bootstrap(primary[primary.seed>=15]);ci.to_csv(path/"paired_intervals.csv",index=False)
    scenarios=[]
    test=data[data.seed>=15]
    grid=[] if args.reuse_sensitivity else itertools.product(PROFILES,[10.,50.,100.],[10000.,20000.,40000.],[1,10,100,1000],[1.,10.,60.,300.,3600.],[0.,1.,10.,100.],["allocated","marginal"])
    for i,(profile,lw,qw,reuse,h,mig,boundary) in enumerate(grid):
        scenarios.append(summarize(evaluate(test,profile,lw,qw,reuse,h,mig,boundary)))
        if i%1000==0:print("scenario",i,flush=True)
    sensitivity=pd.read_csv(path/"sensitivity.csv") if args.reuse_sensitivity else pd.concat(scenarios,ignore_index=True)
    if not args.reuse_sensitivity:sensitivity.to_csv(path/"sensitivity.csv",index=False)
    marginal=[]
    for fraction in [.001,.01,.1]:
        cfg={**DEFAULT,"boundary":"marginal","marginal_fraction":fraction}
        marginal.append(summarize(evaluate(test,**cfg)))
    pd.concat(marginal).to_csv(path/"marginal_fraction_sensitivity.csv",index=False)
    policy=policies(data,path,primary)
    policy.groupby("policy",as_index=False).agg(invocation_rate=("invoked_quantum","mean"),useful=("useful_rate","mean"),compute_j=("compute_j","mean"),total_j=("total_j","mean"),saving_vs_milp_j=("saving_vs_milp_j","mean")).to_csv(path/"policy_summary.csv",index=False)
    # Explicit diagnostic lifetime crossings; late candidates cannot be deployed.
    be=[]
    for r in primary[(primary.seed>=15)&primary.algorithm.str.contains("qaoa|onehot")].itertuples():
        for comparator in ["milp","heuristic"]:
            c=primary[(primary["size"]==r.size)&(primary.seed==r.seed)&(primary.algorithm==comparator)].iloc[0]
            pq=r.feasible_power+(1-r.feasible_rate)*r.incumbent_w;pc=c.feasible_power
            numerator=r.compute_j-c.compute_j+r.transition_j-c.transition_j+(r.delay_s-c.delay_s)*(r.incumbent_w-pc)
            eligible=r.on_time and c.on_time and pc>pq
            lifetime=max(r.delay_s,c.delay_s,r.delay_s+numerator/(pc-pq)) if eligible else np.nan
            be.append({"size":r.size,"seed":r.seed,"algorithm":r.algorithm,"shots":r.shots,"comparator":comparator,
                       "break_even_horizon_s":lifetime,"reason":"finite conditional crossing" if eligible else "no deployable positive power improvement"})
    pd.DataFrame(be).to_csv(path/"break_even.csv",index=False)
    legacy_reanalysis(path)
    extra_diagnostics(path,data,primary)
    dest=figures(path,primary,sensitivity,policy)
    quantum=summary[summary.algorithm.str.contains("qaoa|onehot")]
    keys={"primary_scenario":DEFAULT,"instance_count":len(data[["size","seed"]].drop_duplicates()),
          "test_instances":len(test[["size","seed"]].drop_duplicates()),"primary":summary.replace({np.inf:None}).to_dict("records"),
          "paired_intervals":ci.to_dict("records"),"scenario_count":4320,
          "positive_quantum_scenario_means_vs_milp":int(((sensitivity.algorithm.str.contains("qaoa|onehot"))&(sensitivity.saving_vs_milp_j>1e-8)).sum()),
          "quantum_primary_timely_range":[float(quantum.on_time.min()),float(quantum.on_time.max())],
          "policy":pd.read_csv(path/"policy_summary.csv").to_dict("records"),
          "analysis_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (path/"key_results.json").write_text(json.dumps(keys,indent=2,allow_nan=False)+"\n")
    # Manuscript table and macros are generated, never hand-entered numbers.
    chosen=[("milp",0),("local_search",0),("heuristic",0),("qaoa_online",1024),("qaoa_transfer",64),("qaoa_transfer",1024),("onehot_transfer",64),("onehot_transfer",1024)]
    lines=[r"\begin{tabular}{lrrrrr}",r"\toprule",r"Pipeline & Shots & Quality (\%) & Timely (\%) & Compute (J) & $\Delta E$ (J)\\",r"\midrule"]
    for alg,shots in chosen:
        r=summary[(summary.algorithm==alg)&(summary.shots==shots)].iloc[0]
        excess=0. if abs(r.saving_vs_milp_j)<1e-12 else -r.saving_vs_milp_j
        lines.append(f"{LABELS[alg]} & {shots or '--'} & {r.quality*100:.1f} & {r.on_time*100:.1f} & {r.compute_j:,.2f} & {excess:,.2f} \\\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"]);(dest/"principal_results.tex").write_text("\n".join(lines)+"\n")
    macros=[]
    for name,alg,shots in [("Milp","milp",0),("Local","local_search",0),("Greedy","heuristic",0),("Online","qaoa_online",1024),("Transfer","qaoa_transfer",64),("Onehot","onehot_transfer",64)]:
        r=summary[(summary.algorithm==alg)&(summary.shots==shots)].iloc[0]
        for suffix,value in [("Compute",r.compute_j),("Extra",-r.saving_vs_milp_j),("Quality",100*r.quality),("Timely",100*r.on_time),("Useful",100*r.useful)]:
            if abs(value)<1e-12:value=0.
            macros.append(f"\\newcommand{{\\Energy{name}{suffix}}}{{{value:.2f}}}")
    (dest/"energy_numbers.tex").write_text("\n".join(macros)+"\n")
    print(summary[["algorithm","shots","quality","on_time","compute_j","saving_vs_milp_j"]].to_string(index=False),flush=True)
    print("outputs",dest,flush=True)

if __name__=="__main__":main()
