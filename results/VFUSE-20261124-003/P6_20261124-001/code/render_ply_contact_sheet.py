#!/usr/bin/env python3
"""Render RGB-colored aggregate PLY comparisons without changing point geometry."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=Path(__file__).resolve().parents[2]
PLY=Path(__file__).resolve().parents[1]/'results/pointclouds/predicted_camera'
OUT=Path(__file__).resolve().parents[1]/'results/visualizations'
ALL_SCENES=['office','pipes','courtyard','delivery_area','kicker']
VARIANTS=[('omega_raw','Omega raw'),('vfuse_c1_constant','V-FUSE C1 constant'),('vfuse_c2_minmax','V-FUSE C2 min–max'),('vfuse_c3_percentile','V-FUSE C3 percentile'),('vfuse_c4_rank','V-FUSE C4 rank/CDF')]
DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('confidence','<f4'),('source_view','u1')],align=False)
BG='white'
def read_ply(path):
 with path.open('rb') as f:
  while True:
   line=f.readline()
   if not line: raise ValueError(f'bad PLY header {path}')
   if line.startswith(b'element vertex '): n=int(line.split()[2])
   if line==b'end_header\n':break
  return np.fromfile(f,dtype=DTYPE,count=n)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scene',choices=ALL_SCENES+['all'],default='all');args=ap.parse_args()
 scenes=ALL_SCENES if args.scene=='all' else [args.scene];rng=np.random.default_rng(20261124)
 fig=plt.figure(figsize=(19,19 if len(scenes)>1 else 5.3),facecolor=BG)
 for ri,scene in enumerate(scenes):
  clouds=[read_ply(PLY/scene/'aggregate_refs_01_04'/f'{key}.ply') for key,_ in VARIANTS]
  base=clouds[0];sample=base[rng.choice(len(base),min(60000,len(base)),replace=False)]
  center=np.array([np.median(sample[k]) for k in ('x','y','z')]);xyz=np.column_stack([sample['x'],sample['y'],sample['z']])-center
  eigval,eigvec=np.linalg.eigh(np.cov(xyz,rowvar=False));basis=eigvec[:,np.argsort(eigval)[::-1]]
  for j in range(3):
   k=np.argmax(np.abs(basis[:,j]))
   if basis[k,j]<0:basis[:,j]*=-1
  transformed=[];ranges=[]
  for cloud in clouds:
   d=cloud[rng.choice(len(cloud),min(160000 if len(scenes)==1 else 90000,len(cloud)),replace=False)]
   pos=(np.column_stack([d['x'],d['y'],d['z']])-center)@basis
   rgb=np.column_stack([d['red'],d['green'],d['blue']]).astype(np.float32)/255
   transformed.append((pos,rgb));ranges.append(np.quantile(pos,[.01,.99],axis=0))
  low=np.min([r[0] for r in ranges],axis=0);high=np.max([r[1] for r in ranges],axis=0);mid=(low+high)/2;half=float(np.max(high-low))/2;low=mid-half;high=mid+half
  for ci,((pos,rgb),(_,label)) in enumerate(zip(transformed,VARIANTS)):
   ax=fig.add_subplot(len(scenes),len(VARIANTS),ri*len(VARIANTS)+ci+1,projection='3d',facecolor=BG)
   ax.scatter(pos[:,0],pos[:,1],pos[:,2],c=rgb,s=.5 if len(scenes)==1 else .32,alpha=.88,linewidths=0,depthshade=False,rasterized=True)
   ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_zlim(low[2],high[2]);ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev=24,azim=-55);ax.set_axis_off()
   if ri==0:ax.set_title(label,fontsize=11,pad=2,color='black')
   if ci==0:ax.text2D(-.12,.5,scene,transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=11,fontweight='bold',color='black')
 fig.suptitle('ETH3D held-out point clouds — RGB view',fontsize=16,y=.995,color='black')
 fig.text(.5,.006,'Same predicted-camera frame and ref-0 scale across columns. PCA rotation is display-only; point sampling is for PNG rendering.',ha='center',fontsize=9,color='black')
 fig.subplots_adjust(left=.055,right=.995,top=.96,bottom=.035,wspace=.015,hspace=.015)
 OUT.mkdir(parents=True,exist_ok=True);name='aggregate_rgb_comparison_detail.png' if args.scene=='all' else f'{args.scene}_comparison_detail.png';dest=OUT/name
 fig.savefig(dest,dpi=190,bbox_inches='tight',facecolor=BG);plt.close(fig);print(dest)
if __name__=='__main__':main()
