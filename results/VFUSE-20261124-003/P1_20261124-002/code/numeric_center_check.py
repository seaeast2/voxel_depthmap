#!/usr/bin/env python3
"""Numerically evaluate the ETH3D thin-prism projection at each optical center."""
import json, math
from pathlib import Path
RUN=Path(__file__).resolve().parents[2]
P1=RUN/'P1_20261124-001'; OUT=Path(__file__).resolve().parents[1]/'results'
def project_center(frame):
 u=frame['camera']['params']; d=frame['original_camera']['params']
 fx,fy,cx,cy=u[:4]; fdx,fdy,cdx,cdy=d[:4]; k1,k2,p1,p2,k3,k4,sx1,sy1=d[4:12]
 x=(cx-cx)/fx; y=(cy-cy)/fy; r=math.sqrt(x*x+y*y); theta=math.atan(r); r2=theta*theta
 radial=1+k1*r2+k2*r2**2+k3*r2**3+k4*r2**4
 invr=theta/r if r>1e-10 else 1.0; xd=x*invr; yd=y*invr
 xdn=xd*radial+2*p1*xd*yd+p2*(r2+2*xd*xd)+sx1*r2
 ydn=yd*radial+2*p2*xd*yd+p1*(r2+2*yd*yd)+sy1*r2
 mapped=[fdx*xdn+cdx,fdy*ydn+cdy]; expected=[cdx,cdy]
 return {'frame':frame['name'],'undistorted_center':[cx,cy],'distorted_center_expected':expected,'projected_center':mapped,'numerical_error_px':math.dist(mapped,expected)}
def main():
 scenes=[]
 for scene in ['office','pipes','courtyard','delivery_area','kicker']:
  manifest=json.loads((P1/f'results/frames/{scene}.json').read_text())
  checks=[project_center(f) for f in manifest['frames']]
  scenes.append({'scene':scene,'checks':checks,'max_numerical_error_px':max(c['numerical_error_px'] for c in checks),'pass_tol_1e-6_px':all(c['numerical_error_px']<1e-6 for c in checks)})
 out={'run_id':'VFUSE-20261124-003','stage_id':'P1_20261124-002','purpose':'clarify the center check reported by P1_20261124-001','test_scope':'numerical evaluation of the COLMAP THIN_PRISM_FISHEYE equation exactly at the PINHOLE optical center; this is not an image-wide inverse round-trip/remap validation','scenes':scenes}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'center_projection_check.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
