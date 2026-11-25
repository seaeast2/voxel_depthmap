#!/usr/bin/env python3
"""Reference-0-only scale calibration; held-out ETH3D depth metrics and confidence ranking diagnostics."""
import csv, json, math
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parents[2]
P1=RUN/'P1_20261124-001'; P2=RUN/'P2_20261124-001'; P3=RUN/'P3_20261124-004'; P4=RUN/'P4_20261124-001'; OUT=Path(__file__).resolve().parents[1]/'results'
SCENES=['office','pipes','courtyard','delivery_area','kicker']
CONDITIONS=['c1_constant','c2_minmax','c3_percentile','c4_rank']
def rankdata(x):
 order=np.argsort(x,kind='mergesort'); ranks=np.empty(len(x),np.float64); ranks[order]=np.arange(len(x),dtype=np.float64)
 i=0
 while i<len(order):
  j=i+1
  while j<len(order) and x[order[j]]==x[order[i]]: j+=1
  ranks[order[i:j]]=(i+j-1)/2
  i=j
 return ranks
def rho(a,b):
 if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0: return None
 x=rankdata(a); y=rankdata(b); return float(np.corrcoef(x,y)[0,1])
def metrics(gt,pred,gtvalid):
 good=gtvalid & np.isfinite(pred) & (pred>0)
 n_gt=int(gtvalid.sum()); n=int(good.sum())
 if not n: return {'gt_valid_pixels':n_gt,'valid_prediction_pixels':0,'coverage':0.,'mae':None,'rmse':None,'abs_rel':None}
 e=pred[good]-gt[good]; ae=np.abs(e); ar=ae/np.maximum(gt[good],1e-8)
 return {'gt_valid_pixels':n_gt,'valid_prediction_pixels':n,'coverage':float(n/n_gt),'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'abs_rel':float(ar.mean())}
def diag(conf, pred, gt, gtvalid):
 good=gtvalid & np.isfinite(pred)&(pred>0)&np.isfinite(conf)
 c=conf[good].astype(np.float64); err=np.abs(pred[good]-gt[good])/np.maximum(gt[good],1e-8)
 if len(c)==0:return None,[],[]
 order=np.argsort(c,kind='mergesort'); chunks=np.array_split(order,10); bins=[]
 for k,ix in enumerate(chunks):
  bins.append({'confidence_decile_ascending':k+1,'count':int(len(ix)),'confidence_min':float(c[ix].min()),'confidence_max':float(c[ix].max()),'median_abs_rel':float(np.median(err[ix])),'mean_abs_rel':float(np.mean(err[ix]))})
 risk=[]
 descending=np.argsort(-c,kind='mergesort')
 for coverage in range(10,101,10):
  n=max(1,int(math.ceil(len(c)*coverage/100))) ; ix=descending[:n]
  risk.append({'coverage_percent':coverage,'count':int(n),'mean_abs_rel':float(np.mean(err[ix])),'median_abs_rel':float(np.median(err[ix]))})
 summary={'valid_pixels':int(len(c)),'spearman_confidence_vs_abs_rel':rho(c,err),'confidence_mean':float(np.mean(c)),'confidence_std':float(np.std(c))}
 return summary,bins,risk
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[]; drows=[]; rrows=[]; scales={}
 for scene in SCENES:
  with np.load(P2/f'results/omega_raw/{scene}/vggt_omega_raw.npz') as z:
   raw=z['depth'][0,...,0].astype(np.float32); rawconf=z['depth_conf'][0].astype(np.float32)
  with np.load(P1/f'results/gt_pinhole/{scene}/gt_depth_undistorted.npz') as z:
   gt=z['depth'].astype(np.float32); gv=z['valid'].astype(bool)
  fit=gv[0]&np.isfinite(raw[0])&(raw[0]>0); scale=float(np.median(gt[0][fit]/raw[0][fit])); scales[scene]={'scale_gt_over_omega_ref0_median':scale,'fit_reference_index':0,'fit_pixel_count':int(fit.sum())}
  methods=[('omega_raw',None)] + [(x,x) for x in CONDITIONS] + [('c3_gtcamera','gtcamera')]
  for ref in [1,2,3,4]:
   base=raw[ref]*scale
   preds={'omega_raw':(base,rawconf[ref])}
   for cond in CONDITIONS:
    p=P3/f'results/inference/{scene}/{cond}/ref_{ref:02d}/v_fuse.npz'
    with np.load(p) as z: preds[cond]=(z['fused_depth'].astype(np.float32)*scale,z['fused_conf'].astype(np.float32))
   p=P4/f'results/inference/{scene}/c3_gtcamera/ref_{ref:02d}/v_fuse.npz'
   with np.load(p) as z: preds['c3_gtcamera']=(z['fused_depth'].astype(np.float32)*scale,z['fused_conf'].astype(np.float32))
   for method,(pred,conf) in preds.items():
    m=metrics(gt[ref],pred,gv[ref]); row={'scene':scene,'reference_index':ref,'method':method,'scale_fit_reference_index':0,'scale_gt_over_omega':scale,**m}
    rows.append(row); ds,bins,risk=diag(conf,pred,gt[ref],gv[ref]); drows.append({'scene':scene,'reference_index':ref,'method':method,**ds})
    for b in bins:drows.append({'scene':scene,'reference_index':ref,'method':method,**b})
    for q in risk:rrows.append({'scene':scene,'reference_index':ref,'method':method,**q})
 with (OUT/'heldout_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 with (OUT/'confidence_diagnostics.csv').open('w',newline='') as f:
  fields=sorted({k for r in drows for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(drows)
 with (OUT/'risk_coverage.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rrows[0])); w.writeheader(); w.writerows(rrows)
 # aggregate with per-frame means, explicitly equal-weight across held-out frames
 grouped={}
 for r in rows:grouped.setdefault(r['method'],[]).append(r)
 aggregate={m:{k:(float(np.mean([x[k] for x in rr if x[k] is not None])) if any(x[k] is not None for x in rr) else None) for k in ['coverage','mae','rmse','abs_rel']} | {'n_heldout_frames':len(rr)} for m,rr in grouped.items()}
 summary={'run_id':'VFUSE-20261124-003','stage_id':'P5_20261124-001','status':'completed','scenes':SCENES,'evaluation_reference_indices':[1,2,3,4],'scale_fit_reference_index':0,'gt_domain':'ETH3D laser depth remapped into undistorted PINHOLE output grid; sparse valid mask','prediction_scale':'same scene scale estimated from reference 0 and applied to every prediction including V-FUSE outputs','aggregation':'equal-weight mean across 20 held-out frames; pixel metrics computed within each frame','confidence_diagnostic':'rank/calibration proxy; Spearman and decile error plus descending-confidence risk-coverage. Confidence is not a calibrated probability.','scale_fits':scales,'mean_metrics_by_method':aggregate,'rows':len(rows)}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
