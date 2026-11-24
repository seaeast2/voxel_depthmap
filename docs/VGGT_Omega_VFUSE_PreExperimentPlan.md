# VGGT-Omega + V-FUSE 사전 실험 계획 (Pre-experiment)

- 작성일: 2026-11-24
- 상태: 계획 초안 (실험 미수행)
- 목적: Voxel 기반 깊이 보정 본 연구에 앞서 기존 V-FUSE의 적용 가능성과 비교 기준 확보
- 저장소: seaeast2/voxel_depthmap

## 1. 연구 배경과 범위

VGGT-Omega가 연속 RGB 영상에서 추정한 Depth Map, Confidence Map 및 카메라 파라미터를 V-FUSE에 입력하여 깊이 정확도 및 다중 시점 기하 정합성이 개선되는지 확인한다. V-FUSE는 기존 MVS 출력에 대해 학습된 깊이 융합 모델이며, 기준 카메라 frustum에서 Support, Occlusion, Free-space Violation을 표현하는 Visibility Constraint Volume(VCV)을 사용한다. 본 연구에서 검토하는 world-space voxel grid와는 다른 표현이다.

이번 사전 실험은 두 모델의 공식 pretrained checkpoint를 고정하고, 데이터 변환과 평가만 구현한다. 자체 voxel 알고리즘, V-FUSE 재학습, 실시간 처리, Gaussian 생성은 후속 단계로 둔다. 결과가 부정적이어도 유효한 연구 결과로 기록하며, 연동 성공 자체를 독창성으로 주장하지 않는다.

## 2. 연구 질문 및 가설

| ID | 연구 질문 | 검증 가설 |
|---|---|---|
| RQ1 | 두 모델의 입력/출력 구조가 호환되는가? | 좌표계, 깊이 스케일, 해상도 및 confidence 변환으로 추론 가능하다. |
| RQ2 | 기존 가중치로 깊이 오차를 줄일 수 있는가? | 일부 장면에서 raw VGGT-Omega 대비 MAE/AbsRel이 감소한다. |
| RQ3 | confidence 분포 차이가 성능에 영향을 주는가? | raw, 정규화, 균일 confidence 간 보정 성능이 다르다. |
| RQ4 | 추정 pose 오차가 보정 효과를 제한하는가? | GT 카메라 조건에서 성능이 개선될 수 있다. |
| RQ5 | 기하학적 정합성 향상이 실제 정확도 향상으로 이어지는가? | GT 오차와 cross-view consistency를 별도로 평가해야 한다. |

## 3. 참고 자료 및 구현 검증

- V-FUSE: Burgdorfer & Mordohai, ICCV 2023, 특히 §3.1 VCV, §3.2 depth regression, §3.3 SWE, §5 실험.
  - https://openaccess.thecvf.com/content/ICCV2023/html/Burgdorfer_V-FUSE_Volumetric_Depth_Map_Fusion_with_Long-Range_Constraints_ICCV_2023_paper.html
  - https://github.com/nburgdorfer/V-FUSE
- VGGT-Omega: https://github.com/facebookresearch/vggt-omega
  - 공식 모델: https://huggingface.co/facebook/VGGT-Omega
- MVSNet: Yao et al., ECCV 2018, §3.3 confidence.
- 후속 연구 비교: KinectFusion, Voxgraph, NeuralRecon, PointVGGT.

위 논문에 근거한 개념 설계이며 실제 upstream 코드/체크포인트의 세부 API, 라이선스, 카메라 파일 포맷과 정규화 규칙은 P0에서 확인한다. 문서의 경로와 CLI는 제안 사항으로, 현재 구현되어 있음을 뜻하지 않는다.

## 4. 시스템 구성

```text
RGB sequence / selected keyframes
  -> VGGT-Omega inference (frozen)
  -> cached RGB, depth, confidence, K, T_wc
  -> adapter + source view selection
  -> V-FUSE inference (frozen)
  -> refined depth / confidence
  -> backprojection / point cloud
  -> evaluation against raw baseline and GT
```

여기서 T_wc는 본 문서에서 world-to-camera 변환으로 정의한다. 실제 코드의 명명 규칙과 혼동하지 않도록 메타데이터에 convention을 명시한다.

### 4.1 환경

- Ubuntu, NVIDIA CUDA GPU, Python/PyTorch.
- VGGT-Omega와 V-FUSE의 의존성 충돌을 피하기 위해 별도 가상환경을 우선 사용한다.
- 12GB급 GPU에서는 작은 해상도 및 3~5개 view로 smoke test를 시작하고 peak VRAM을 측정한다. 두 모델은 순차 실행하고 중간 산출물을 디스크에 저장한다. CUDA OOM 발생 시 view 수/해상도를 낮추되 intrinsics를 함께 변환한다.
- upstream commit SHA, checkpoint 이름 및 checksum, Python/PyTorch/CUDA 버전, GPU, seed, 데이터셋 버전을 manifest에 기록한다.
- VGGT-Omega 모델 접근 승인 및 checkpoint 확보 여부를 확인한다. 접근 불가 시 P0 blocked로 표시하고 다른 모델 결과를 Omega 결과로 대체하지 않는다.
- 원본 모델 코드 및 가중치의 재배포 라이선스를 확인하고 가중치, 토큰, 데이터셋 원본은 Git에 커밋하지 않는다.

### 4.2 데이터 계약

| 항목 | 내부 표준 | 검증 |
|---|---|---|
| RGB | N×H×W×3, uint8 또는 명시적 float 범위 | 모델 전처리와 동일한 좌표 |
| Depth | N×H×W, float32 | 양의 camera Z-depth, NaN/Inf/invalid mask |
| Confidence | N×H×W, float32 | raw 보존, 변환 방식 및 범위 기록 |
| Intrinsics K | N×3×3 | resize/crop 반영 |
| Extrinsics E | N×4×4 | world-to-camera, 마지막 행 [0,0,0,1] |
| Valid mask | N×H×W | 원본/보정 공통 평가용 |
| View IDs | N | 순서, reference/source 매핑 |
| Depth bounds | reference별 min/max | upstream parser 단위 및 탐색 범위 확인 |

Omega 공식 API의 depth, depth_conf, pose encoding 및 전처리된 이미지 추출 경로를 P0에서 확인한다. batch 및 singleton channel 축을 명시적으로 변환한다. V-FUSE 핵심 네트워크는 RGB를 직접 feature matching에 사용하는 것이 아니라 깊이/신뢰도와 카메라 기하를 사용한다. RGB는 데이터 로더, 시각화 및 point cloud 색상에 필요할 수 있다.

V-FUSE용 PFM(depth/confidence), camera TXT, PNG, reference/source pair 목록 또는 custom dataset을 작성한다. 공식 loader의 실제 디렉터리, depth interval, endian, PFM 상하 반전, dtype, view count, tensor shape를 확인한 후 export/import round-trip 테스트를 수행한다. 유효하지 않은 깊이의 처리 방식은 upstream mask 규칙을 따른다.

### 4.3 기하학적 변환

동차 픽셀 p=[u,v,1]^T, Z-depth d에 대해 X_c=d K^-1 p. World-to-camera E_i=[R_i|t_i]이면 X_w=R_i^T(X_c-t_i), 다른 view j에서 X_j=R_j X_w+t_j, p_j=pi(K_j X_j).

- depth가 ray length라면 Z-depth로 변환해야 한다. 공식 API/공식 unprojection 결과와 비교한다.
- resize/crop 변환 A에 대해 K'=AK를 사용한다. pixel-center convention도 검증한다.
- 전역 scale s 적용 시 depth와 translation을 함께 s배 하고 회전은 유지한다. 단위가 없는 예측에 임의의 미터 단위를 붙이지 않는다.
- 독립 window 간 좌표/scale 정합은 이번 범위 밖이다. 초기 실험은 장면별 단일 inference window로 제한한다.
- GT 비교 시 동일한 scene-level similarity alignment를 raw와 refined 모두에 적용한다. scale은 raw 결과에서 한 번 산출하여 고정하고, 픽셀별 또는 방법별 재정합으로 오차를 숨기지 않는다.
- GT pose 대조 실험에서는 GT와 predicted depth의 scale을 먼저 공통 단위로 맞춘다. 필요하면 GT K와 GT E를 함께 사용하는 별도 조건을 기록한다. GT 기반 scale 사용은 oracle 평가 조건으로 명시한다.

### 4.4 Confidence 변환

유효 픽셀에 대한 raw confidence 분포(min, max, quantile, histogram)를 저장한다. V-FUSE는 VCV의 각 제약 응답을 confidence로 가중하고 SWE에서도 confidence 통계를 사용하므로, 단순 범위 변환이 확률적 의미의 동등성을 보장하지 않는다.

| ID | 변환 | 정의 |
|---|---|---|
| C0 | Raw | upstream이 허용하는 경우 그대로 사용 |
| C1 | Constant | 유효 픽셀 1, invalid 0 |
| C2 | Min-max | (c-min)/(max-min+eps) |
| C3 | Percentile | clip((c-Q05)/(Q95-Q05+eps),0,1) |
| C4 | Rank | 유효 픽셀의 경험적 CDF (선택 실험) |

통계는 기본적으로 장면 전체 유효 픽셀에서 구하고 모든 view에 동일 매핑을 적용한다. per-view 정규화는 상대 신뢰도 왜곡 가능성이 있어 별도 ablation으로만 수행한다. 분모가 0인 경우 C1로 fallback하며 기록한다. C0가 loader의 허용 범위를 위반하면 실행하지 않고 incompatible로 기록한다.

### 4.5 View selection 및 depth bounds

- Reference는 고정된 keyframe 목록을 사용한다. 처음에는 reference 1개 + source 2~4개.
- 시간 인접 프레임을 기본으로 하되 추정 카메라 baseline과 관측 overlap을 확인한다. pure rotation, 낮은 overlap, 극단적 baseline은 별도 기록한다.
- baseline과 보정 모델 모두 동일한 reference 및 입력 frame set을 사용한다.
- depth bounds는 유효 depth의 robust quantile을 초기 후보로 사용하고 여유 폭을 둔다. 공식 SWE 구현의 min/max 및 hypothesis 수 설정을 우선 준수한다.
- GT가 search window 밖에 존재하는 비율을 진단용으로 측정하되 GT를 일반 입력의 bounds 결정에 사용하지 않는다.
- checkpoint별 학습 depth 단위/범위, normalization을 확인한다. 스케일 민감도가 있으면 전체 depth와 translation을 동시에 변환한 조건으로 비교한다.

## 5. 실험 데이터

| 구분 | 데이터 | 용도 |
|---|---|---|
| Smoke | 정적 장면 1개, 3~5 view | 포맷/재투영/추론 확인 |
| Pilot | ScanNet 또는 7-Scenes 정적 실내 3개 장면, 장면당 5~10 reference | RGB 기반 추정 깊이의 GT 평가 |
| Control | V-FUSE 공식 예제 또는 지원 MVS 데이터 1개 | upstream 정상 실행 확인 |
| Optional | DTU evaluation subset | V-FUSE 학습 도메인에 가까운 비교 |

모델 입력은 RGB만 사용한다. RGB-D 데이터셋의 센서 depth와 pose는 평가 및 명시적으로 구분된 oracle 실험에만 사용한다. GT depth의 invalid/occlusion mask, RGB-depth registration 및 해상도를 확인한다. 장면 선택과 frame stride를 결과 확인 전에 고정하고 manifest로 기록한다. 결과가 좋은 장면만 선별하지 않는다.

## 6. 단계별 실험

| 단계 | 작업 | 산출물 및 통과 조건 |
|---|---|---|
| P0 | 공식 코드/환경/가중치 확인, V-FUSE 예제 실행, Omega 단독 추론 | 버전 manifest, raw prediction cache, 정상 추론 로그. 실패 시 원인 기록 |
| P1 | Adapter 구현, PFM/TXT round-trip, 합성 평면 및 실제 장면 재투영 | shape/finite 검사, 합성 identity 및 known transform reprojection 오차 < 0.1 px 목표 |
| P2 | frozen V-FUSE zero-shot smoke | 유효 depth/confidence 출력, 수치 및 시각 검증, OOM/NaN 여부 |
| P3 | pilot: baseline 및 confidence ablation | 동일 reference에 대한 장면별 지표, error map, confidence histogram |
| P4 | pose 및 scale 민감도 | predicted camera vs GT camera 조건, global scale ablation |
| P5 | 결과 분석 및 본 연구 진입 판단 | report.md, metrics.csv, 실패 사례, 후속 연구 권고 |

P0~P2가 실패하면 환경, 좌표계, scale, confidence, loader, SWE bounds 순서로 진단하고 변경 사항을 기록한다. P3/P4는 P2 통과 후 시행한다. fine-tuning은 별도 실험 계획으로 분리한다.

### 6.1 비교군

| ID | 조건 |
|---|---|
| B0 | VGGT-Omega raw depth |
| B1 | VGGT-Omega + 단순 cross-view filtering (선택, 제거 효과 비교) |
| F0 | VGGT-Omega + frozen V-FUSE + raw confidence (허용 시) |
| F1 | VGGT-Omega + frozen V-FUSE + constant confidence |
| F2 | VGGT-Omega + frozen V-FUSE + min-max confidence |
| F3 | VGGT-Omega + frozen V-FUSE + percentile confidence |
| O1 | F3와 동일하되 GT camera 조건 (oracle) |
| O0 | B0의 GT camera 기반 point cloud (O1 비교용, depth 자체는 동일) |

모든 조건에서 동일한 checkpoint, 원본 depth, frame 목록, 평가 mask 및 alignment를 유지한다. 기본 체크포인트는 공식 제공 모델 중 하나로 사전 지정하고 나머지는 선택 실험으로 둔다. confidence ablation에서는 confidence 변환 외의 설정을 고정한다.

## 7. 평가 방법

### 7.1 2D Depth

- MAE = mean(|d_pred-d_gt|), RMSE = sqrt(mean((d_pred-d_gt)^2)), AbsRel = mean(|d_pred-d_gt|/d_gt).
- GT valid 및 raw/refined 공통 valid mask에서 비교하고, 전체 GT valid 대비 coverage를 별도로 보고한다. invalid output을 누락하여 지표만 개선하는 경우를 방지한다.
- 전역 metric scale을 확보하지 못하면 scene-level alignment 적용 후의 지표임을 명시한다. 절대 metric 정확도와 구분한다.
- raw/refined depth, absolute error, difference, confidence, valid mask 시각화.

### 7.2 다중 시점 정합성

reference depth를 source view에 재투영하여 source depth와 비교한다. 동일한 camera, 공통 overlap, z-buffer/occlusion 판정 규칙 및 경계 mask를 사용한다. 상대 깊이 residual 및 median/mean, 관측 수를 기록한다. GT 정확도와 독립적으로 보고하며 잘못된 공통 geometry도 높은 consistency를 가질 수 있음을 명시한다.

### 7.3 3D 평가

동일한 카메라 및 sampling/fusion 조건으로 point cloud를 생성한다. GT point cloud/mesh 사용 가능 시 accuracy, completeness, Chamfer distance, F-score, outlier ratio를 평가한다. outlier는 GT 표면과의 거리 > 사전 지정 임계값으로 정의한다(예: metric indoor 기준 2cm/5cm, 데이터 특성에 맞게 고정). GT 가시 영역을 기준으로 평가하고 point 수 및 coverage를 함께 기록한다. GT가 없으면 outlier 감소를 확정하지 않는다.

### 7.4 효율 및 결과 집계

- VGGT 추론, export, V-FUSE, point cloud 및 전체 시간 분리. CUDA 동기화 및 warm-up 후 반복 측정.
- peak allocated/reserved VRAM, input resolution, view count, depth hypotheses.
- 장면별/프레임별 결과 및 장면 단위 macro average, 개선/악화 장면 수.
- 개선율 = (baseline error - refined error)/baseline error × 100%.
- 소규모 pilot 결과를 통계적 일반화로 해석하지 않는다.

## 8. 사전 실험 판단 기준

아래 기준은 논문 성능을 인용한 것이 아니라 본 프로젝트의 초기 의사결정 기준이다.

- 기술적 성공: 공식 예제 및 Omega 입력으로 V-FUSE 추론 성공, geometry/format 검증 통과, 반복 가능한 실행.
- 긍정적 신호: pilot macro AbsRel 또는 MAE 5% 이상 상대 개선, coverage 감소 2 percentage points 이내, 다수 장면에서 개선. RMSE와 3D 지표에 현저한 악화가 없는지 함께 확인한다.
- 혼합 결과: consistency만 개선되거나 coverage가 크게 감소하면 기하학적 정확도 향상으로 결론 내리지 않는다.
- 부정적 결과: scale/confidence/pose를 통제해도 개선이 없거나 악화하면 원인을 분석하고 V-FUSE를 비교군으로 유지할지 결정한다.
- 본 연구 진입: world-space voxel 기반 방법을 설계할 때 V-FUSE의 reference-frustum 표현, 학습 의존성, 반복 view별 처리 및 pose 오차 민감도와 비교한다. 단순 차이만으로 novelty를 확정하지 않는다.

## 9. 구현 및 저장 구조 (제안)

```text
docs/
  VGGT_Omega_VFUSE_PreExperimentPlan.md
  experiments/vggt_omega_vfuse/<run_id>/report.md
experiments/vggt_omega_vfuse/
  configs/
  scripts/
    infer_omega.py
    export_vfuse.py
    validate_geometry.py
    run_vfuse.py
    evaluate.py
  tests/
  README.md
outputs/vggt_omega_vfuse/<run_id>/
  manifest.json
  raw/
  vfuse_input/
  refined/
  metrics.csv
  visualizations/
  logs/
```

실제 repository 구조에 맞춰 경로를 조정한다. raw 데이터, checkpoint, 대용량 PFM/PLY 및 비공개 데이터는 gitignore 처리하고 소스, 설정, 작은 요약 지표, 보고서만 커밋한다. 각 단계는 독립적으로 재실행할 수 있도록 설계한다. 중간 cache는 입력/checkpoint/config hash를 키로 사용한다.

## 10. 보고서 템플릿

- Run ID / 날짜 / 실행자 / commit SHA
- 데이터셋, scene/frame IDs, 모델 checkpoint, 하드웨어 및 환경
- 실험 조건, scale 변환, confidence 방식, view selection, depth bounds
- P0~P5 상태(pass/fail/blocked/not run)
- 정량 결과: B0 대비 MAE/RMSE/AbsRel/coverage/consistency/3D 지표
- 정성 결과: raw/refined depth, point cloud, 오류 사례
- 실행 시간, peak VRAM, 오류 로그
- 결론: 적용 가능성, domain shift, pose/scale 영향, 다음 실험
- 재현 명령 및 산출물 경로

## 11. 후속 연구 방향

사전 실험 결과를 바탕으로 자체 world-space voxel 보정 알고리즘을 설계한다. V-FUSE의 Support/Occlusion/Free-space Violation을 관측 증거로 참고하되, 공간에 고정된 voxel에 관측 정보를 누적하는 방식과 학습 없는 깊이 보정의 효과를 검증한다. 최종 3D Gaussian Splatting에 대한 영향은 별도 통제 실험에서 측정한다.

이번 문서는 계획이며 성능 결과나 실제 구현 성공을 주장하지 않는다.
