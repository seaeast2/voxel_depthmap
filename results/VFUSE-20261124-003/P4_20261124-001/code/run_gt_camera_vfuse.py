#!/usr/bin/env python3
"""V-FUSE oracle camera ablation; uses reference-0 GT only to scale depth/poses."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parents[2]
P0=RUN/'P0_20261124-001'; P1=RUN/'P1_20261124-001'; P2=RUN/'P2_20261124-001'; STAGE=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--scene',required=True,choices=['office','pipes','courtyard','delivery_area','kicker']); a=ap.parse_args()
 import torch, yaml
 sys.path.insert(0,str(P0/'assets/source/V-FUSE'))
 from src.models.fusion import Fusion
 from src.utils.common import warp
 scene=a.scene; out=STAGE/f'results/inference/{scene}/c3_gtcamera'
 if out.exists(): raise FileExistsError(out)
 frames=json.loads((P1/f'results/frames/{scene}.json').read_text())['frames']
 raw_path=P2/f'results/omega_raw/{scene}/vggt_omega_raw.npz'
 gt_path=P1/f'results/gt_pinhole/{scene}/gt_depth_undistorted.npz'
 with np.load(raw_path) as z:
  depth=z['depth'][0,...,0].astype(np.float32); raw_conf=z['depth_conf'][0].astype(np.float32); images=z['images'][0].astype(np.float32)
 with np.load(gt_path) as z: gt=z['depth'].astype(np.float32); gvalid=z['valid'].astype(bool)
 n,h,w=depth.shape
 if n!=5 or len(frames)!=5 or gt.shape!=depth.shape: raise ValueError('frozen 5-view inputs disagree')
 valid=np.isfinite(depth)&(depth>0)&np.isfinite(raw_conf)
 scale_values=gt[0][gvalid[0]&valid[0]]/depth[0][gvalid[0]&valid[0]]
 scale=float(np.median(scale_values))
 if not np.isfinite(scale) or scale<=0: raise ValueError('invalid reference-0-only scale')
 q05,q95=np.quantile(depth[valid],[.05,.95]); pad=.1*(q95-q05); near=max(float(np.finfo(np.float32).eps),float(q05-pad)); far=float(q95+pad)
 conf=np.where(valid,np.maximum(np.clip((raw_conf-np.quantile(raw_conf[valid],.05))/max(np.quantile(raw_conf[valid],.95)-np.quantile(raw_conf[valid],.05),1e-12),0,1),1e-4),0).astype(np.float32)
 cams=np.zeros((5,2,4,4),np.float32)
 for i,f in enumerate(frames):
  K=f['camera']; iw,ih=K['width'],K['height']; sx=w/iw; sy=h/ih
  fx,fy,cx,cy=K['params'][:4]
  ext=np.asarray(f['world_to_camera'],np.float32).copy(); ext[:,3]/=scale
  cams[i,0,:3,:4]=ext; cams[i,0,3,3]=1
  cams[i,1,0,0]=fx*sx; cams[i,1,1,1]=fy*sy
  cams[i,1,0,2]=(cx+.5)*sx-.5; cams[i,1,1,2]=(cy+.5)*sy-.5; cams[i,1,2,2]=1; cams[i,1,3]=[near,0,256,far]
 cfg=yaml.safe_load((P0/'assets/source/V-FUSE/configs/DTU/dtu.yaml').read_text()); cfg['device']='cuda:0'; cfg['eval']['num_frame']=5; cfg['eval']['depth_planes']=8; cfg['eval']['scale']=1.; cfg['swenet']['depth_mode']='conf'
 torch.manual_seed(0); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); model=Fusion(cfg,8).to('cuda:0')
 checkpoint=P0/'assets/checkpoints/vfuse_gbinet_V10.0.7.pt'; model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True); model.eval()
 out.mkdir(parents=True,exist_ok=False); records=[]
 for ref in range(5):
  order=[(ref+i)%5 for i in range(5)]; dd=depth[order]; cc=conf[order]; cam=cams[order].copy()
  dt=torch.from_numpy(dd[:,None]); ct=torch.from_numpy(cc[:,None]); camt=torch.from_numpy(cam)
  wd,wc=warp(dt,ct,camt.clone()); data={'depths':wd.unsqueeze(1).unsqueeze(0),'orig_depths':dt.unsqueeze(0),'confs':wc.unsqueeze(1).unsqueeze(0),'orig_confs':ct.unsqueeze(0),'images':torch.from_numpy(images[order]).unsqueeze(0),'cams':camt.unsqueeze(0),'num_frame':5,'scene':scene,'ref_index':ref}
  data={k:(v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k,v in data.items()}
  torch.cuda.synchronize(); t=time.perf_counter()
  with torch.inference_mode(): o=model(data)
  torch.cuda.synchronize(); elapsed=time.perf_counter()-t
  fd=o['fused_depth'][0,0].float().cpu().numpy(); fc=o['fused_conf'][0,0].float().cpu().numpy(); bounds=o['depth_bounds'].float().cpu().numpy()
  if not np.isfinite(fd).all() or not np.isfinite(fc).all(): raise ValueError(f'nonfinite oracle fusion scene={scene} ref={ref}')
  dest=out/f'ref_{ref:02d}'; dest.mkdir(parents=True,exist_ok=False); npz=dest/'v_fuse.npz'; np.savez_compressed(npz,fused_depth=fd,fused_conf=fc,depth_bounds=bounds)
  rec={'reference_index':ref,'frame':frames[ref]['name'],'ordered_frame_ids':[frames[i]['name'] for i in order],'status':'completed','nonpositive_fused_pixel_count':int((fd<=0).sum()),'seconds':elapsed,'output_npz':str(npz),'output_npz_sha256':sha(npz)}; (dest/'metrics.json').write_text(json.dumps({'run_id':'VFUSE-20261124-003','stage_id':'P4_20261124-001','scene':scene,'condition':'c3_percentile','camera_source':'ETH3D undistorted PINHOLE calibration + COLMAP pose','scale_fit_reference_index':0,'scale_median_gt_over_omega_ref0':scale,'scale_fit_pixel_count':int(scale_values.size),'translation_conversion':'GT metric camera translations divided by ref-0 depth scale to match Omega depth units','intrinsics_resize':'pixel-center convention: (c+0.5)*scale-0.5; focal multiplied by per-axis resize','gt_depth_used_for_reference0_scale':True,'heldout_gt_depth_read_by_inference':False,'confidence_floor':1e-4,'checkpoint_sha256':sha(checkpoint),'input_npz_sha256':sha(raw_path),**rec,'gpu':torch.cuda.get_device_name()},indent=2)+'\n'); records.append(rec); del data,o; torch.cuda.empty_cache(); print(f'completed={scene}/c3_gtcamera/ref{ref:02d} seconds={elapsed:.3f}',flush=True)
 summary={'run_id':'VFUSE-20261124-003','stage_id':'P4_20261124-001','scene':scene,'status':'completed','nonpositive_fused_pixel_count':sum(r['nonpositive_fused_pixel_count'] for r in records),'camera_source':'ETH3D GT PINHOLE intrinsics and COLMAP poses','scale_fit_reference_index':0,'scale_median_gt_over_omega_ref0':scale,'scale_fit_pixel_count':int(scale_values.size),'evaluation_reference_indices':[1,2,3,4],'conditions':['c3_percentile'],'inference_count':len(records),'runs':records,'gt_depth_used_for_reference0_scale':True,'heldout_gt_depth_read_by_inference':False,'gpu':torch.cuda.get_device_name()}; (out/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__': main()
