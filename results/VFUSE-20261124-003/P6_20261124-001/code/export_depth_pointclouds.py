#!/usr/bin/env python3
"""Export paired color PLYs for frozen held-out Omega/V-FUSE depth maps."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parents[2]
P1=RUN/'P1_20261124-001'; P2=RUN/'P2_20261124-001'; P3=RUN/'P3_20261124-004'; P4=RUN/'P4_20261124-001'; P5=RUN/'P5_20261124-002'; STAGE=Path(__file__).resolve().parents[1]
SCENES=['office','pipes','courtyard','delivery_area','kicker']; REFS=[1,2,3,4]; CONDS=['c1_constant','c2_minmax','c3_percentile','c4_rank']
DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('confidence','<f4'),('source_view','u1')],align=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_ply(path, points):
 path.parent.mkdir(parents=True,exist_ok=True)
 header=('ply\nformat binary_little_endian 1.0\ncomment RGB-colored depth point cloud; confidence is a scalar field\n'
         f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nproperty uchar source_view\nend_header\n').encode('ascii')
 with path.open('wb') as f: f.write(header); points.tofile(f)
 return {'path':str(path),'point_count':int(len(points)),'size_bytes':path.stat().st_size,'sha256':sha(path)}
def pointcloud(depth,conf,image,K,world_to_camera,scale,view_id):
 h,w=depth.shape; yy,xx=np.indices((h,w),dtype=np.float32); fx=float(K[0,0]); fy=float(K[1,1]); cx=float(K[0,2]); cy=float(K[1,2])
 valid=np.isfinite(depth)&(depth>0)&np.isfinite(conf)&np.isfinite(image).all(axis=0)
 z=depth[valid].astype(np.float64)*scale; x=(xx[valid].astype(np.float64)-cx)/fx*z; y=(yy[valid].astype(np.float64)-cy)/fy*z
 xyz_cam=np.stack([x,y,z,np.ones_like(z)],axis=1)
 ext=np.eye(4,dtype=np.float64); ext[:3,:4]=world_to_camera
 c2w=np.linalg.inv(ext); xyz_world=xyz_cam@c2w.T; rgb=np.clip(np.rint(image[:,valid].T*255),0,255).astype(np.uint8)
 pts=np.empty(len(z),dtype=DTYPE); pts['x']=xyz_world[:,0];pts['y']=xyz_world[:,1];pts['z']=xyz_world[:,2];pts['red']=rgb[:,0];pts['green']=rgb[:,1];pts['blue']=rgb[:,2];pts['confidence']=conf[valid];pts['source_view']=view_id
 return pts

def main():
 if (STAGE/'results/pointclouds').exists(): raise FileExistsError(f'refusing to overwrite P6 point clouds: {STAGE}/results/pointclouds')
 STAGE.mkdir(parents=True,exist_ok=True); out=STAGE/'results/pointclouds'; out.mkdir(parents=True,exist_ok=True)
 report=json.loads((P5/'results/summary.json').read_text()); scale_map={s:float(v['scale_gt_over_omega_ref0_median']) for s,v in report['scale_fits'].items()}
 all_records=[]; scene_summaries=[]
 for scene in SCENES:
  with np.load(P2/f'results/omega_raw/{scene}/vggt_omega_raw.npz') as z:
   raw=z['depth'][0,...,0].astype(np.float32); rawconf=z['depth_conf'][0].astype(np.float32); ext=z['extrinsics'][0].astype(np.float32); K=z['intrinsics'][0].astype(np.float32); images=z['images'][0].astype(np.float32)
  frames=json.loads((P1/f'results/frames/{scene}.json').read_text())['frames']; scale=scale_map[scene]
  for ref in REFS:
   viewdir=out/'predicted_camera'/scene/f'ref_{ref:02d}'; pair=[]; clouds=[]
   variants=[('omega_raw',raw[ref],rawconf[ref])]
   for cond in CONDS:
    with np.load(P3/f'results/inference/{scene}/{cond}/ref_{ref:02d}/v_fuse.npz') as z: variants.append((f'vfuse_{cond}',z['fused_depth'].astype(np.float32),z['fused_conf'].astype(np.float32)))
   for label,depth,conf in variants:
    pts=pointcloud(depth,conf,images[ref],K[ref],ext[ref],scale,ref)
    rec=write_ply(viewdir/f'{label}.ply',pts)
    meta={'run_id':'VFUSE-20261124-003','stage_id':'P6_20261124-001','scene':scene,'reference_index':ref,'frame_id':frames[ref]['name'],'variant':label,'camera_source':'VGGT-Omega predicted world-to-camera and intrinsics','coordinate_frame':'VGGT-Omega predicted scene frame, scaled by reference-0 GT/Omega median depth ratio','metric_scale_gt_over_omega':scale,'depth_source':'P2 raw Omega map' if label=='omega_raw' else f'P3 {label.removeprefix("vfuse_")} fused depth','color_source':'resized RGB image from P2 cache','confidence_scalar_source':'Omega depth_conf' if label=='omega_raw' else 'V-FUSE fused_conf','invalid_or_nonpositive_depth_excluded':int((~np.isfinite(depth)|(depth<=0)|~np.isfinite(conf)).sum()),**rec}
    (viewdir/f'{label}.json').write_text(json.dumps(meta,indent=2)+'\n'); all_records.append(meta);pair.append(rec);clouds.append(pts)
   # Scene-wide four-reference clouds are assembled below; source_view enables view filtering.
  aggregate_records=[]
  for vi,(label,_,_) in enumerate(variants):
   parts=[]
   for ref in REFS:
    with np.load(P2/f'results/omega_raw/{scene}/vggt_omega_raw.npz') as z:
     # Reconstruct metadata arrays once per reference for the aggregate to keep the implementation simple and deterministic.
     raw=z['depth'][0,...,0].astype(np.float32); rawconf=z['depth_conf'][0].astype(np.float32); ext=z['extrinsics'][0].astype(np.float32); K=z['intrinsics'][0].astype(np.float32); images=z['images'][0].astype(np.float32)
    if vi==0: depth,conf=raw[ref],rawconf[ref]
    else:
     cond=CONDS[vi-1]
     with np.load(P3/f'results/inference/{scene}/{cond}/ref_{ref:02d}/v_fuse.npz') as z:depth=z['fused_depth'].astype(np.float32);conf=z['fused_conf'].astype(np.float32)
    parts.append(pointcloud(depth,conf,images[ref],K[ref],ext[ref],scale,ref))
   rec=write_ply(out/'predicted_camera'/scene/'aggregate_refs_01_04'/f'{label}.ply',np.concatenate(parts))
   rec.update({'scene':scene,'variant':label,'reference_indices':REFS,'coordinate_frame':'VGGT-Omega predicted scene frame scaled by the scene reference-0 scale','metric_scale_gt_over_omega':scale}); aggregate_records.append(rec)
  scene_summaries.append({'scene':scene,'scale_fit_reference_index':0,'scale_gt_over_omega':scale,'heldout_references':REFS,'per_reference_clouds':len(REFS)*len(variants),'aggregate_clouds':len(aggregate_records),'aggregates':aggregate_records})
  print(f'exported_scene={scene} heldout_plys={len(REFS)*len(variants)} aggregate_plys={len(aggregate_records)}',flush=True)
 # P4 outputs are in the ETH3D/COLMAP metric world frame and remain separate from predicted-camera PLYs.
 oracle=[]
 for scene in SCENES:
  manifest=json.loads((P1/f'results/frames/{scene}.json').read_text());frames=manifest['frames'];scale=scale_map[scene]
  with np.load(P2/f'results/omega_raw/{scene}/vggt_omega_raw.npz') as z:images=z['images'][0].astype(np.float32)
  for ref in REFS:
   f=frames[ref];cam=f['camera'];ih,iw=cam['height'],cam['width']; h,w=images.shape[-2:]; sx=w/iw;sy=h/ih; fx,fy,cx,cy=cam['params'][:4]
   kg=np.array([[fx*sx,0,(cx+.5)*sx-.5],[0,fy*sy,(cy+.5)*sy-.5],[0,0,1]],np.float32); ext=np.asarray(f['world_to_camera'],np.float32)
   with np.load(P4/f'results/inference/{scene}/c3_gtcamera/ref_{ref:02d}/v_fuse.npz') as z:depth=z['fused_depth'].astype(np.float32);conf=z['fused_conf'].astype(np.float32)
   pts=pointcloud(depth,conf,images[ref],kg,ext,scale,ref); path=out/'gt_camera_oracle'/scene/f'ref_{ref:02d}'/'vfuse_c3_percentile.ply'; rec=write_ply(path,pts)
   meta={'run_id':'VFUSE-20261124-003','stage_id':'P6_20261124-001','scene':scene,'reference_index':ref,'frame_id':f['name'],'variant':'vfuse_c3_percentile_gtcamera','camera_source':'ETH3D undistorted PINHOLE intrinsics and COLMAP poses','coordinate_frame':'ETH3D/COLMAP world frame; metric camera translations and ref-0 scale applied to depth','metric_scale_gt_over_omega':scale,'depth_source':'P4 GT-camera C3 fused depth','color_source':'resized RGB image from P2 cache','confidence_scalar_source':'P4 fused_conf','invalid_or_nonpositive_depth_excluded':int((~np.isfinite(depth)|(depth<=0)|~np.isfinite(conf)).sum()),**rec}
   (path.parent/'vfuse_c3_percentile.json').write_text(json.dumps(meta,indent=2)+'\n');oracle.append(meta)
 summary={'run_id':'VFUSE-20261124-003','stage_id':'P6_20261124-001','status':'completed','format':'binary_little_endian PLY','properties':['x','y','z','red','green','blue','confidence','source_view'],'main_comparison':'raw Omega and P3 C1-C4 use identical per-reference VGGT-Omega camera poses/intrinsics and scale, so their PLYs are directly comparable in that predicted scene frame','heldout_reference_indices':REFS,'predicted_camera_per_reference_ply_count':len(all_records),'predicted_camera_scene_aggregate_ply_count':sum(x['aggregate_clouds'] for x in scene_summaries),'oracle_gtcamera_ply_count':len(oracle),'oracle_coordinate_frame_note':'P4 oracle-camera PLYs use ETH3D/COLMAP world coordinates and are intentionally separate; do not directly overlay with predicted-camera PLYs without a coordinate alignment','scale_fits_from_P5':scale_map,'scenes':scene_summaries,'oracle_plys':oracle}
 (STAGE/'results/pointcloud_export_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'status':'completed','per_reference_plys':len(all_records),'scene_aggregate_plys':summary['predicted_camera_scene_aggregate_ply_count'],'oracle_plys':len(oracle),'output_root':str(out)},indent=2))
if __name__=='__main__':main()
