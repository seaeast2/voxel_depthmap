# P6 PLY point-cloud export

P3에서 비교한 raw Omega 및 V-FUSE C1–C4를 held-out refs 1–4별 컬러 PLY로 변환했다. 같은 scene/ref 안의 5개 파일은 동일한 Omega predicted camera pose/intrinsics, reference-0 scale, RGB를 사용하므로 서로 직접 겹쳐 볼 수 있다.

## 파일 경로

- View별 paired 비교: `results/pointclouds/predicted_camera/<scene>/ref_01..04/omega_raw.ply`, `vfuse_c1_constant.ply` … `vfuse_c4_rank.ply`
- 4개 held-out reference 병합: `results/pointclouds/predicted_camera/<scene>/aggregate_refs_01_04/`
- P4 GT-camera PLY: `results/pointclouds/gt_camera_oracle/<scene>/ref_01..04/vfuse_c3_percentile.ply`. 이 파일은 ETH3D/COLMAP world frame이므로 predicted-camera PLY와 바로 겹치면 안 된다.

PLY에는 XYZ, RGB, `confidence`, `source_view` 속성이 있다. 유한하고 양수인 depth와 유한한 confidence가 있는 pixel만 점으로 저장했다. 깊이는 P5 reference-0 scale로 metric 단위에 맞췄다. P3 PLY들은 Omega의 예측 좌표계에 있고 P4는 ETH3D/COLMAP 좌표계에 있어 P4를 비교할 때는 별도 좌표 정합이 필요하다.

총 145개 PLY (P3 view별 100, P3 scene aggregate 25, P4 oracle 20), 약 724 MB. 포맷은 binary little-endian이다. 대용량 PLY 본체는 stage `.gitignore`에서 제외하고, export 코드·요약·sidecar JSON은 저장한다.

실행: `code/export_depth_pointclouds.py`; 해시와 점 수 목록: `results/pointcloud_export_summary.json` 및 인접 PLY `.json` sidecar.

CloudCompare에서 직접 비교할 때는 같은 장면과 reference의 `omega_raw.ply` 및 `vfuse_c*_*.ply`를 함께 열면 된다. 두 점군은 동일한 predicted-camera 좌표계와 scale을 사용한다. `aggregate_refs_01_04`는 4개 held-out reference를 합친 장면 단위 비교다. confidence scalar를 색상으로 표시하거나 `source_view`로 aggregate의 frame을 필터링할 수 있다.

## 채팅에서 보는 PNG 미리보기

- 전체 5 scene × Omega raw/C1–C4 비교: `results/visualizations/aggregate_rgb_comparison_detail.png`
- delivery_area 확대 비교: `results/visualizations/delivery_area_comparison_detail.png`

PNG 렌더링은 비교 화면의 가독성을 위해 각 PLY에서 점을 균일 무작위 추출했고, 한 장면의 비교 열에 동일한 축 범위를 사용했다. 표시를 위한 PCA 회전은 모든 열에 공통 적용했다. 따라서 PNG는 형태를 빠르게 확인하는 미리보기이며, PLY의 모든 점을 표시한 것은 아니다. 시각적 형태 차이는 정확도 개선의 증거가 아니므로 held-out 수치와 함께 해석해야 한다.
