# 임시 실험 계획 — VGGT-O 깊이의 voxel 기반 정제와 순차 지도 구축

## 목적

VGGT-O가 추정한 깊이와 카메라 포즈를 사용해 다음 세 가지를 검증한다.

1. **제거** — 단일 프레임의 국소 기하로 고립점과 깊이 경계 오류 후보를 찾고, 얇은 구조와 정상 경계가 얼마나 보존되는지 확인한다.
2. **보정** — 여러 시점의 가시성 제약으로 원래 깊이의 불일치와 대안 깊이에 대한 지지가 모두 확인된 픽셀만 보정한다. 맞은편 표면으로 잘못 이동시키거나 기존 오차를 키우는 사례를 측정한다.
3. **지도 구축** — 동일한 포즈와 융합 설정에서 제거와 보정이 누적 지도의 표면 정확도, 완전성, 재방문 안정성에 미치는 효과를 확인한다.

첫 실험 범위는 좌표계 검증(E0)부터 고정 포즈 순차 지도 구축(E3)까지다. 위치 추정과 지도 갱신을 결합하는 SLAM 실험(E4)과 학습 기반 비교(E5)는 후속 단계다. 데이터셋, 모델 버전, 수치 판정 기준은 실행 전에 확정한다.

## 구현에서 사용하는 좌표와 관측

depth map의 픽셀 `(u, v)`는 카메라 광선 하나를 지정하고, 예측 깊이는 그 광선 위에 표면 관측 하나를 놓는다. 해상도는 광선의 영상 내 샘플 간격을 정한다. 깊이는 광선 위의 연속값이며, 깊이 정확도는 별도로 평가한다. 카메라 내부 파라미터와 픽셀 대응을 고정하면 깊이 보정에서 바꾸는 값은 픽셀당 깊이 하나다.

- **픽셀·깊이 후보 격자 `(u, v, k)`** — `k`는 같은 광선 위의 깊이 후보다. 영상 이웃은 깊이 경계나 같은 구조에 속하는지 검토하는 데 쓴다. V-FUSE의 가시성 제약은 이 격자에 놓인다.
- **월드 직교 voxel 격자 `(X, Y, Z)`** — 깊이를 카메라 좌표로 역투영하고 포즈를 적용한 점을 일정 간격의 격자에 배치한다. 3D 이웃에서 표면 지지를 평가하고 여러 프레임을 한 지도에 누적하는 데 쓴다. 같은 voxel에 여러 관측이 들어가도 각각의 픽셀 출처를 유지한다.
- **수정하는 값** — E1은 지도에 사용할 관측을 픽셀 마스크로 선택한다. E2는 선택한 픽셀의 깊이를 바꾸고 그 점의 3D 위치와 voxel 소속을 다시 계산한다. E3는 그 관측들로 voxel의 TSDF와 가중치를 갱신한다. E4는 카메라 포즈를 보정해 해당 프레임의 점들을 함께 이동시킨다.

## 선행 연구에서 학습 모델이 맡는 부분

### V-FUSE

V-FUSE는 여러 뷰의 depth·confidence map과 카메라 기하를 사용해 기준 뷰의 융합 깊이를 추정한다. Search Window Estimation 네트워크가 픽셀별 깊이 탐색 범위를 정한다. 각 깊이 후보에 대한 지지, 가림, 빈 공간 위반을 기하로 계산해 가시성 제약 볼륨에 기록한다. 학습된 3D 합성곱 네트워크가 볼륨을 정규화하고, soft-argmax 연산이 깊이 확률분포를 연속 깊이 값으로 변환한다. 융합 깊이는 GT depth 오차로, 탐색 범위는 GT 포함과 범위 크기에 대한 손실로 학습한다. [V-FUSE 논문](https://openaccess.thecvf.com/content/ICCV2023/papers/Burgdorfer_V-FUSE_Volumetric_Depth_Map_Fusion_with_Long-Range_Constraints_ICCV_2023_paper.pdf)

본 계획에서는 지지·가림·빈 공간 위반을 계산하는 기하 규칙을 E2에 적용한다. 이 규칙이 단순 재투영 기준선보다 유효한지는 별도 비교로 검증한다.

### NeuralRecon

NeuralRecon은 포즈가 주어진 영상 프레임에서 특징을 추출해 3D 볼륨에 투영하고, coarse-to-fine sparse 3D 네트워크로 local fragment의 TSDF와 occupancy를 예측한다. GRU 기반 모듈은 현재 fragment의 특징을 이전 전역 상태와 융합한다. 훈련에는 TSDF와 occupancy 정답을 사용한다. 비교 대상은 순차 표면 재구성과 융합이다. [NeuralRecon 논문](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_NeuralRecon_Real-Time_Coherent_3D_Reconstruction_From_Monocular_Video_CVPR_2021_paper.pdf)

## 실험 단계

### E0 — 입력과 좌표계 검증

- VGGT-O의 공식 모델명, 저장소·커밋, 체크포인트, 추론 명령, 프레임 목록 및 시간, 해상도와 resize/crop/padding을 확인한다. 모델의 confidence 출력 정의와 저장·시각화 단계의 후처리를 기록한다. 깊이 단위와 camera-Z/ray distance 구분, intrinsics의 기준 격자, extrinsics 방향과 축 규약을 기록한다.
- 알려진 카메라와 3D 점으로 픽셀 ↔ 카메라 ↔ 월드 변환을 검증한다. 서로 다른 두 카메라의 알려진 대응점으로 포즈 방향, 깊이 규약과 재투영 오차를 확인한다. 동일한 변환의 왕복 검사와 별도로 수행한다.
- 단일 이미지 추론, 다중 이미지 공동 추론, 구간별 추론 중 사용한 방식을 기록한다. 구간별 추론을 쓸 경우 겹치는 프레임을 이용해 좌표계와 스케일을 공통 공간에 연결할 수 있는지 확인한다. 미터 스케일의 근거가 없으면 모델 단위로 기록하고, GT 비교에 사용할 스케일 보정과 정렬 방식은 평가 전에 고정한다.
- 원본 깊이를 voxel에 투영하고 `(frame_id, pixel_u, pixel_v)`에서 3D 점 및 voxel까지 추적할 수 있게 기록한다. 서로 다른 프레임의 관측이 동일 voxel에 들어가도 각각의 출처를 보존한다.

**산출물과 판정.** 변환 규약, 테스트 입력과 잔차, 프레임별 픽셀·voxel 대응, 구간 정렬의 근거를 저장한다. 좌표 변환 테스트가 사전 허용 오차를 충족하고, 실제 입력의 좌표계·단위·프레임 대응과 원본 픽셀 출처가 확인되면 E1·E2로 진행한다. 구간 간 정렬을 검증할 수 없으면 지도 구축 범위를 동일 좌표계의 구간으로 제한한다.

### E1 — 단일 프레임의 불량점 후보 검출과 제거

- 프레임 내 깊이 불연속, 공간 고립, 거리별 점 밀도, 국소 표면 잔차와 법선 일관성을 후보 특징으로 삼는다. `surface-supported`, `unsupported`, `uncertain`으로 태깅한다. 단일 프레임으로 얻는 표면 지지 점수는 제거 후보의 근거로 사용한다. 휘어진 벽처럼 주변과 일관되게 틀리는 깊이는 E2에서 다시 평가한다.
- 같은 봉처럼 이어지는 구조라는 근거가 있는 픽셀은 관측 voxel이 떨어져 있어도 `uncertain`으로 남긴다. 양쪽 관측이 하나의 구조를 지지하고 중간 점만 벗어나는 경우는 깊이 보정 후보로 기록할 수 있다. E1에서는 해당 점의 깊이를 유지한다.
- 카메라 광선이 평면과 거의 평행하게 지나가면 이웃 픽셀에서 역투영한 점 사이의 간격이 벌어져, 연속된 실제 평면도 voxel 격자에서 끊겨 보일 수 있다. 이런 평면 후보에서 유효 관측 수와 국소 표면 지지가 사전 기준에 미달하면 `unsupported`로 표시하고, E1의 voxel 필터 결과에서 해당 관측을 제외한다. 제거 사유는 관측 지지 부족으로 기록하고 원본 깊이와 픽셀 대응은 보존한다.
- 같은 입력으로 원본, confidence 단독 필터, depth-only `edge_3x3_3pct`·`edge_3x3_5pct`, voxel 표면 지지 필터를 비교한다. 두 depth-only 기준선은 유효한 3×3 이웃의 깊이 범위를 중심 깊이의 3% 또는 5%와 비교하며 confidence를 사용하지 않는다. 방법별 제거량이 다르면 유사한 보존율에서의 성능도 비교한다. 입력 깊이가 다중 이미지 공동 추론 결과일 경우 이를 명시하고, 검출 규칙이 사용한 정보의 범위를 한 프레임으로 제한한다.
- RGB/depth 위에 삭제 후보를 표시한다. 경계와 얇은 구조를 사전에 지정한 ROI에서 확인하고, GT depth가 있으면 고오차 점 검출률과 정상 점 오제거율을 함께 보고한다. 표면 소속 오류는 별도의 소속 라벨이 있는 사례에서 평가한다.

**산출물과 판정.** 원본 및 필터별 PLY, 픽셀별 지지 점수·상태·판정 사유, 제거·보존 마스크와 픽셀 대응을 저장한다. 기준선 대비 불량점 검출과 구조 보존을 함께 판단하고, 카메라 광선과 거의 평행한 평면 ROI에서 제거율과 GT가 뒷받침하는 정상 관측의 손실을 따로 기록한다. E2는 E1 결과와 독립적으로 원본 깊이에서 검증한다.

### E2 — 다중 시점 합의에 의한 깊이 보정

- 대상 프레임을 제외한 소스 프레임만으로 참조 표면을 만든다. 원래 깊이가 유효한 소스 관측과 사전 허용 오차를 넘어 불일치하고, 서로 다른 시점이 같은 대안 표면을 충분히 지지할 때 보정한다. 대안 깊이는 대상 픽셀의 카메라 광선 위에서 선택하고, 픽셀 위치는 유지한다. voxel 연결성만으로 생긴 후보는 원래 깊이를 유지한다.
- 먼저 원래 깊이 후보의 다중 시점 지지와 모순을 검사한다. V-FUSE를 참고한 기하 기준선에서는 소스 관측을 대상 뷰에 재투영하고, 재투영 깊이가 후보에 가까울수록 높은 지지 점수를 준다. 처음에는 기하 규칙으로 평가하고, confidence를 지지 가중치에 쓰는 경우 그 효과를 별도로 비교한다. [V-FUSE §3.1](https://nburgdorfer.github.io/vfuse/static/pdfs/main.pdf#page=3)
- 시야 밖, 무효 깊이, 가림, 동적 물체 의심, 시차 부족, 경계 혼합을 구분한다. 소스 시점에서 더 가까운 표면에 가려진 후보는 관측 불가로 처리한다. 관측 가능한 빈 공간을 침범하는 후보에는 위반을 기록한다. 소스 프레임 수, 최소 시차와 깊이 일치 허용 오차를 사전에 정하고 같은 뷰의 관측을 중복 집계하지 않는다. 상충하는 표면 후보가 있거나 지지가 부족하면 원래 값을 보존하고 사유를 기록한다.
- 원본 깊이, 동일 소스 뷰를 이용한 단순 재투영 합의, voxel 기반 보정을 비교한다. 같은 포즈와 소스 선택으로 voxel 표현의 추가 효과를 평가한다. GT 또는 검증된 기준 포즈가 있으면 동일 예측 깊이를 기준 포즈와 예측 포즈 조건에서 비교해 포즈 오차의 영향을 진단한다. GT 포즈와 결합할 때 깊이 스케일을 맞추는 절차와 그 정보 출처를 명시한다.
- 합의에 사용한 뷰와 평가용 뷰를 분리한다. 공동 추론으로 이미 정보를 공유한 뷰의 오차는 관측 일관성 지표로 보고한다. 독립된 뷰의 평가를 주장할 경우 해당 프레임을 보정용 추론·정렬·참조 표면 구성에서 제외한다. 고의로 삽입한 이상치로 복구 동작을 확인할 수 있으며 실제 예측 오차에 대한 평가는 별도로 수행한다.

**산출물과 판정.** 원본 및 보정 깊이, 픽셀별 이전·이후 voxel 소속과 보정·유지 사유, 지지·가림·위반 관측과 소스 프레임 ID를 저장한다. GT가 있으면 공통 유효 픽셀의 깊이 오차와 보정 대상 중 개선·악화된 픽셀 비율을 함께 비교한다. 경계·얇은 구조의 악화와 보정 적용률도 확인한다. GT가 없으면 별도 평가 뷰에서의 재투영 일관성과 보정 실패 사례까지 보고한다.

### E3 — 고정 포즈 순차 voxel 지도 구축

- KinectFusion식 기하 융합을 기준선으로 구현한다. 월드 voxel 중심을 각 프레임에 투영하고 해당 관측 깊이와 voxel 깊이의 차이로 TSDF를 계산한다. 표면 앞의 양수 거리값은 상한에서 자르고, 표면 근처 뒤쪽은 음수로 누적한다. 관측 깊이보다 허용 범위 이상 뒤인 위치는 해당 프레임으로 갱신하지 않는다. voxel마다 거리값의 가중 평균과 누적 가중치를 유지하고, 표면은 TSDF의 0 교차에서 보간해 추출한다. 같은 방향의 깊이 오차가 여러 관측에 반복되면 융합 표면에도 남을 수 있다. [KinectFusion §3.3–3.4](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf#page=5)
- E1 제거 마스크는 3D 표면점 출력과 TSDF 입력에 모두 적용한다. 제외된 픽셀에서는 표면과 표면 앞 빈 공간 모두 갱신을 건너뛴다. 유효 관측이 한 번도 기여하지 않은 voxel은 미관측으로 유지한다.
- 동일한 프레임 순서, 포즈, voxel 크기, TSDF truncation 범위, 가중치 및 표면 추출 조건으로 단순 TSDF 융합을 비교한다. 원본, 제거만, 보정만, 제거와 보정의 결합을 비교해 각 처리의 기여를 확인한다. 결합 조건은 보정 후 제거 여부를 결정하는 등 처리 순서를 사전에 고정하고, 보정된 관측과 실제 제외된 관측을 기록한다.
- 광선과 평면이 거의 평행한 관측에 낮은 가중치를 주는 KinectFusion식 융합은, E1의 제거 자체가 주는 효과를 확인한 뒤 별도 비교 조건으로 검토한다. [KinectFusion §3.3](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf#page=5)
- 짧은 구간과 재방문을 포함한 긴 구간에서 누적 지도를 평가한다. GT 포즈가 있으면 예측 포즈와 교체한 진단을 수행한다. 없으면 고정 시드의 포즈 교란으로 민감도를 측정한다. 두 경우 모두 원본과 처리 결과에 같은 포즈 조건을 적용한다.
- 부분 지도와 최종 지도를 저장한다. 3D GT가 있으면 공통 관측 영역에서 accuracy/completeness를 측정하고, 해당 GT에 적용 가능한 경우 F-score와 거리 임계값을 함께 기록한다. 이중 표면·표면 두께, 재방문에서의 변화량, 처리 시간, peak 메모리, 활성 voxel 수를 함께 보고한다. 향후 프레임을 사용해 깊이 또는 포즈를 얻었다면 전체 실험을 오프라인 재생으로 기록한다.

**완료 기준.** E3의 결론은 주어진 포즈에서의 순차 지도 구축 성능으로 한정한다. 제거로 생긴 결손과 보정으로 생긴 표면 오차를 함께 비교하고, 지도 개선 여부를 E1·E2의 국소 결과와 별도로 판정한다.

### E4 — 후속 SLAM 실험

E3에서 융합과 포즈 민감도를 확인한 뒤, **고정 지도에서 한 프레임의 포즈를 보정하는 실험**부터 시작한다. 대상 프레임을 빼고 만든 참조 지도에 동일한 VGGT-O 초기 포즈와 대상 깊이를 사용한다. 같은 정합 알고리즘으로 원본 깊이 지도와 E1 필터 적용 지도를 비교해, 불량점 제거가 포즈 보정에 미치는 영향을 확인한다. E4 실행 전에 참조·대상 프레임, 공통 지도 스케일, 유효 대응 및 포즈 갱신 판정 기준을 정한다.

- **거리장 직접 정합** — Bylow 방식으로 현재 프레임의 깊이 점을 월드로 옮긴 위치에서 voxel 거리장을 보간하고, 그 SDF 값이 0에 가까워지도록 현재 포즈의 회전과 이동을 최적화한다. [Bylow et al., §III-A–B](https://www.ipb.uni-bonn.de/projects/mod/literatur/pdf/p5-Bylow-2013-SDF.pdf#page=3)
- **표면 정합 비교** — KinectFusion 방식으로 동일한 지도에서 raycasting한 표면점과 법선을 얻고, point-to-plane ICP로 현재 프레임을 정합한다. 벽이 지배적인 구간은 벽을 따라 움직이는 방향처럼 정합 제약이 약한 자유도와 유효 대응 수를 확인한다. 정합 잔차와 GT 포즈 오차를 함께 보고한다. [KinectFusion §3.4–3.5](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf#page=6)
- **여러 뷰의 포즈 보정 후보** — PointVGGT의 voxel 해싱은 뷰 간 대응점 검색에 참고한다. 논문의 포즈 보정은 3D 점 간 거리와 깊이 재투영 오차를 사용하며 센서 깊이를 입력으로 받는다. VGGT-O 예측 깊이만 사용하는 경우는 별도 조건으로 검증한다. 겹치는 voxel 서브맵의 포즈를 직접 맞추는 단계에는 Voxgraph의 거리장 잔차를 참고한다. [PointVGGT §3.4](https://arxiv.org/html/2610.11612v1#S3.SS4), [Voxgraph §VII-C](https://arxiv.org/abs/2004.13154)

이후에는 현재까지 들어온 프레임으로 위치를 추정하고 지도를 갱신하는 순차 조건을 실험한다. 루프 검출과 포즈 최적화를 적용하면 수정된 포즈로 관측을 재통합하거나 서브맵을 재배치한다. 동일 입력에서 E3의 고정 포즈 지도, 포즈 보정 후 지도 및 루프 최적화 전후를 비교한다. GT 궤적이 있으면 정렬 규약을 고정하고 ATE·RPE를 기록하며, 지도 품질과 단계별 지연, peak 메모리를 함께 평가한다. 온라인 또는 실시간 성능의 판정에는 실제 프레임 도착 순서와 모델 추론부터 지도 갱신까지의 처리 시간을 사용한다.

### E5 — 후속 학습 기반 비교

E2의 실패가 깊이 후보 선택에 집중되면 V-FUSE식 깊이 정제를, E3에서 fragment 간 표면 누적 자체가 병목이면 NeuralRecon식 표면 예측·융합을 검토한다. 공개 체크포인트 평가와 재학습을 구분하고, 재학습 시 학습·검증·평가 장면을 분리한다. 입력 뷰, 포즈 출처, 학습 GT, 출력 표현과 온라인 조건을 기록한다. V-FUSE식 방법은 깊이 지표에서, NeuralRecon식 방법은 3D 지도 지표에서 각각 기하 기준선과 비교한다.

단일 프레임 제거에도 3D CNN을 후보로 둘 수 있다. 월드 또는 카메라 기준의 직교 voxel 격자에 점유 여부, 관측 점 수, confidence를 입력하고 관측 voxel의 제거 점수를 예측하는 방식이다. E1에서는 기하 필터를 먼저 평가하고, 학습 모델은 후속 비교에서 다룬다.

## 보류 과제 — 얇고 기울어진 구조의 voxel 안티앨리어싱

voxel 간격보다 얇은 봉이나 기울어진 선은 실제로 이어져 있어도 픽셀 샘플링과 voxel 양자화로 점유가 끊겨 보일 수 있다. 향후에는 관측을 인접 voxel에 가중 분배하는 안티앨리어싱으로 표현상의 단절이 줄어드는지 검증한다. 실제 단절이나 가림까지 잘못 이어 붙이는지 확인하고, 점유 표현의 연결성과 GT 기준 깊이 오차를 각각 평가한다.

현재 실험에서는 이런 단절을 불확실 사례로 기록한다. 안티앨리어싱에 의한 점유 변경과 단절을 메우기 위한 깊이 보정은 후속 실험으로 보류한다. 같은 구조로 볼 근거가 충분한 경우, 의심점을 제외한 양쪽 관측으로 국소 중심선을 추정하고 해당 픽셀 광선에 가깝게 놓이는 깊이 후보를 구하는 방법을 검토할 수 있다.

## 보류 과제 — 관측 지지가 부족한 평면 영역의 재건

E1에서 제외된 저지지 평면 영역은 현재 실험의 필터 결과에서 결손으로 남긴다. 추가 관측이나 별도 최적화로 이 영역을 재건하는 방법은 후속 과제로 둔다. 후속 실험에서는 재건된 영역의 완전성과 잘못 추가된 표면을 함께 평가한다.

## 공통 평가 원칙

- 실제 예측 깊이를 평가할 GT가 있는 정적 다중 뷰 장면을 우선 확보한다. 개발용과 최종 평가용 장면 또는 구간을 사전에 나누고, 얇은 구조, 가림 경계, 저텍스처 표면과 동적 물체는 확인할 ROI로 지정한다.
- 최종 평가 전 프레임 목록, GT 유효 마스크, ROI, 깊이·포즈 정렬 방식, 임계값과 기준선 대비 최소 개선량·허용 손실을 설정에 고정한다. 평가 결과를 보고 설정을 변경한 경우 새 실행 코드에 변경 사유를 남기고 해당 결과를 탐색으로 기록한다.
- 깊이 MAE·RMSE는 동일한 GT 유효 픽셀 중 공통으로 출력된 픽셀에서 비교하고, 전체 GT 유효 픽셀 대비 보존율과 오차 허용 범위 내 유효 픽셀 비율을 함께 보고한다. 제거 수, 깊이 변경 수, 전체 평가 픽셀 수를 별도로 기록한다. 장면별 수치와 전체 집계 방법을 함께 남긴다.
- GT 정렬이 필요하면 동일 입력을 비교하는 모든 방법에 사전에 정한 공통 스케일·정렬을 적용한다. 평가 단위와 정렬에 사용한 데이터를 기록한다. GT 없는 구간의 결론은 관측 일관성, 재방문 안정성, 시각적 변화로 제한한다.

## 실험 코드와 실행 폴더

- 실험 목적에는 단계 코드 `E0`–`E5`를 사용한다. 실제 실행에는 저장소 전체에서 고유한 실행 코드 `R0001`, `R0002` 등을 순서대로 부여한다. 단계 코드와 실행 코드는 모두 영문 대문자 뒤에 숫자가 오는 형식(`^[A-Z]+[0-9]+$`)을 따른다.
- 실행 전에 `runs/<단계 코드>/<실행 코드>/`를 새로 만든다. 예를 들어 `runs/E2/R0007/`이 E2 실행 한 건의 폴더다. 기준선, 비교 조건, 반복 실행, 실패 후 재시도에도 각각 새 코드를 부여한다. 입력이나 설정이 같아도 기존 코드를 재사용하지 않는다.
- 아래 파일과 해당 단계의 산출물을 실행 폴더에 남긴다. 여러 실행의 비교 결과를 새로 생성하는 경우에도 별도 실행 폴더를 만들고 비교에 사용한 실행 코드와 산출물 경로를 기록한다.

```text
runs/E2/R0007/
├── manifest.json   # 단계·실행 코드, 가설, 상태, 명령, 실행 환경, 입력과 이전 실행의 출처, 산출물 목록
├── config.yaml     # 프레임과 추론 설정, 기준선/비교 조건, 평가 마스크·임계값, 시드
├── logs/           # 실행 로그와 실패 시 오류 기록
├── artifacts/      # 원본/보정 깊이, 마스크, PLY, 지도 등 해당 실행의 산출물
├── metrics.json    # 측정값, 평가 대상 수, 적용률과 보존율
└── report.md      # 비교 결과, 판정 근거, 실패 사례, 해석 범위
```

- `manifest.json`에는 데이터 버전, 프레임 ID, 모델·체크포인트·코드 버전, 상위 실행 코드와 참조 파일 경로를 기록한다. 공유 입력이나 대용량 산출물을 참조할 때는 파일 해시 또는 불변 버전도 기록한다. 실행이 실패하면 로그와 부분 산출물을 보존하고 실패 상태 및 중단 지점을 기록한다. 완료된 실행의 결과나 설정은 덮어쓰지 않는다.

## 첫 실행 전에 확정할 항목

- VGGT-O 모델 및 체크포인트, 원본 depth와 confidence의 추출 경로
- GT와 포즈의 가용 범위, 개발·평가 구간, 스케일 및 정렬 절차
- 좌표 검증 허용 오차, voxel 크기, 소스 프레임 선택과 가시성 판정 기준
- E1의 평면별 최소 관측 수와 국소 표면 지지 기준, 카메라 광선과 거의 평행한 평면의 평가 ROI
- 단계별 오차·유효 픽셀 보존율·보정 적용률의 임계값과 최소 개선 기준

## 참고문헌

이 연구에서 새로 참조한 논문은 이 목록 끝에 이어서 추가한다. 저자, 논문 제목, 학회 또는 저널, 발행 연도, 확인 가능한 권·호·쪽수 또는 논문 번호를 기록하고 DOI나 원문 링크를 붙인다. 프리프린트는 arXiv 식별자와 참고한 버전을 기록한다.

1. N. Burgdorfer and P. Mordohai, “V-FUSE: Volumetric Depth Map Fusion with Long-Range Constraints,” *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 3449–3458, 2023. [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Burgdorfer_V-FUSE_Volumetric_Depth_Map_Fusion_with_Long-Range_Constraints_ICCV_2023_paper.html)
2. J. Sun, Y. Xie, L. Chen, X. Zhou, and H. Bao, “NeuralRecon: Real-Time Coherent 3D Reconstruction From Monocular Video,” *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15598–15607, 2021. [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_NeuralRecon_Real-Time_Coherent_3D_Reconstruction_From_Monocular_Video_CVPR_2021_paper.html)
3. R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon, “KinectFusion: Real-Time Dense Surface Mapping and Tracking,” *Proc. IEEE International Symposium on Mixed and Augmented Reality (ISMAR)*, pp. 127–136, 2011. [DOI](https://doi.org/10.1109/ISMAR.2011.6092378)
4. E. Bylow, J. Sturm, C. Kerl, F. Kahl, and D. Cremers, “Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions,” *Proc. Robotics: Science and Systems (RSS)*, 2013. [DOI](https://doi.org/10.15607/RSS.2013.IX.035)
5. W. Huang, H. Gao, X. Lai, H. Liu, and Y. Zhao, “Semantic Fusion and Depth-to-Volume Backend Optimization for Robotic Simultaneous Localization and Mapping,” *Electronics Letters*, vol. 62, no. 1, Art. e70604, 2026. [DOI](https://doi.org/10.1049/ell2.70604)
6. V. Reijgwart, A. Millane, H. Oleynikova, R. Siegwart, C. Cadena, and J. Nieto, “Voxgraph: Globally Consistent, Volumetric Mapping Using Signed Distance Function Submaps,” *IEEE Robotics and Automation Letters*, vol. 5, no. 1, pp. 227–234, 2020. [DOI](https://doi.org/10.1109/LRA.2019.2953859)
7. C. Yuan, W. Xu, X. Liu, X. Hong, and F. Zhang, “Efficient and Probabilistic Adaptive Voxel Mapping for Accurate Online LiDAR Odometry,” *IEEE Robotics and Automation Letters*, vol. 7, no. 3, pp. 8518–8525, 2022. [DOI](https://doi.org/10.1109/LRA.2022.3187250)
8. Z. Liu, H. Li, C. Yuan, X. Liu, J. Lin, R. Li, C. Zheng, B. Zhou, W. Liu, and F. Zhang, “Voxel-SLAM: A Complete, Accurate, and Versatile Light Detection and Ranging-Inertial Simultaneous Localization and Mapping System,” *Advanced Intelligent Systems*, vol. 8, no. 4, Art. e202501081, 2026. [DOI](https://doi.org/10.1002/aisy.202501081)
9. H. Jiang, L. Yu, and J. Zheng, “PointVGGT: Zero-Shot Multiview RGB-D Point Cloud Registration with Visual Geometry Foundation Priors,” *arXiv preprint arXiv:2610.11612v1*, Oct. 2026. [arXiv](https://arxiv.org/html/2610.11612v1)
10. A. Mandal, R. Kumar, S. S. Harithas, and C. Arora, “VGGT-SLAM++,” *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 1008–1018, 2026. [CVF](https://openaccess.thecvf.com/content/CVPR2026W/VOCVALC/html/Mandal_VGGT-SLAM_CVPRW_2026_paper.html)
