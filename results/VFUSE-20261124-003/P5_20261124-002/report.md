# VGGT-Omega + V-FUSE 후속 실험 결과

**Run:** `VFUSE-20261124-003`

**상태:** 주요 추론·평가 완료. 저자 GBiNet train-domain control은 입력 MVS 결과 데이터가 없어 미완료.
**범위:** ETH3D high-res multi-view training split의 office, pipes, courtyard, delivery_area, kicker 다섯 장면. 장면당 RGB만으로 선택한 5 views 중 reference 0은 scale fitting, refs 1–4는 held-out 평가에 썼다.

## 결과 요약

Reference 0에서 구한 장면별 scale을 고정하고 20개 held-out frame의 평균을 계산했다. V-FUSE confidence 변환 조건 네 가지는 이 파일럿에서 raw Omega의 평균 AbsRel을 개선하지 못했다. 이 결과는 작은 scene/frame subset에 대한 진단이며, 일반적인 성능 결론이나 전체 ETH3D benchmark 성능을 뜻하지 않는다.

| 입력/조건 | Coverage | MAE (m) | RMSE (m) | AbsRel |
|---|---:|---:|---:|---:|
| VGGT-Omega raw | 1.000 | 0.0531 | 0.1425 | 0.0132 |
| V-FUSE, C1 constant confidence | 1.000 | 0.1083 | 0.5573 | 0.0160 |
| V-FUSE, C2 global min–max | 1.000 | 0.5380 | 0.8524 | 0.1274 |
| V-FUSE, C3 Q05–Q95 percentile | 1.000 | 0.3635 | 0.7451 | 0.0714 |
| V-FUSE, C4 empirical rank/CDF | 1.000 | 0.2475 | 0.6777 | 0.0520 |
| V-FUSE C3, ETH3D oracle cameras | 1.000 | 0.3671 | 0.7473 | 0.0720 |

Coverage는 held-out sparse GT의 valid pixel 중 유한하고 양수인 예측을 낸 비율이다. P3 전체 fused maps에 0 이하 값 22개가 있었고 모두 delivery_area에서 나왔다. 이 점들은 저장 결과에 보존하고 무효 예측으로 취급했다. 이 22점은 이번 sparse GT 평가 마스크와 겹치지 않아 표의 coverage는 1.0이다.

장면별 AbsRel은 office / pipes / courtyard / delivery_area / kicker 순으로 다음과 같다.

| 조건 | office | pipes | courtyard | delivery_area | kicker |
|---|---:|---:|---:|---:|---:|
| VGGT-Omega raw | 0.0102 | 0.0091 | 0.0119 | 0.0152 | 0.0196 |
| C1 constant | 0.0117 | 0.0134 | 0.0119 | 0.0228 | 0.0199 |
| C2 min–max | 0.1521 | 0.0456 | 0.0821 | 0.2489 | 0.1085 |
| C3 percentile | 0.0479 | 0.0374 | 0.0518 | 0.1411 | 0.0789 |
| C4 rank/CDF | 0.0513 | 0.0319 | 0.0338 | 0.0790 | 0.0642 |
| C3 oracle cameras | 0.0508 | 0.0354 | 0.0506 | 0.1418 | 0.0815 |

장면별 scale은 reference 0의 `median(GT depth / Omega depth)`이며 각각 office 2.2891, pipes 2.3618, courtyard 6.1307, delivery_area 7.7724, kicker 2.6437이다. 이 scale은 모든 held-out raw/V-FUSE 결과에 동일하게 적용했다. Oracle camera 조건의 pose translation도 같은 scale을 기준으로 변환했다.

## Confidence rank 진단

V-FUSE의 `fused_conf`와 픽셀별 absolute-relative error 사이의 장면·frame별 Spearman 상관을 구한 뒤 20개 평가 frame에서 평균했다. 값이 더 음수일수록 높은 confidence가 낮은 오차와 더 강하게 연관되는 순위 경향을 뜻한다.

| 조건의 출력 confidence | 평균 Spearman ρ | 가장 낮은 confidence decile의 평균 frame median AbsRel | 가장 높은 confidence decile의 평균 frame median AbsRel |
|---|---:|---:|---:|
| Omega raw confidence | -0.216 | 0.0198 | 0.0103 |
| V-FUSE C1 | -0.252 | 0.0136 | 0.0071 |
| V-FUSE C2 | -0.570 | 0.1733 | 0.0155 |
| V-FUSE C3 | -0.559 | 0.1571 | 0.0113 |
| V-FUSE C4 | -0.527 | 0.1366 | 0.0112 |
| V-FUSE C3 oracle cameras | -0.597 | 0.1634 | 0.0071 |

순위 진단상 C2–C4 confidence는 오차와 강한 음의 연관을 보였지만, 깊이 지표는 raw Omega보다 나빴다. 따라서 이번 데이터에서는 confidence의 순위 정보와 fused depth 정확도가 분리되어 관찰됐다. 이 수치는 reliability diagram의 확률 보정 오차가 아니며, confidence를 calibrated probability라고 해석하지 않는다. coverage별 오차는 `results/risk_coverage.csv`, frame별 decile은 `results/confidence_diagnostics.csv`에 있다.

## 수행한 후속 권고

1. **추가 ETH3D 장면:** office/pipes 외에 courtyard, delivery_area, kicker를 확보해 같은 RGB-only 기준으로 view를 선택했다. ETH3D 공개 목록에는 각 장면의 undistorted images, calibration 및 depth archive가 제공된다. [ETH3D dataset 목록](https://eth3d.ethz.ch/datasets)
2. **Confidence 대안:** C1 constant, C2 global min–max, C3 global Q05–Q95, C4 scene-global empirical rank/CDF를 모두 평가했다. C2–C4 유효 confidence의 하한은 1e-4로 고정했다. delivery_area C2는 융합 depth에 음수가 생겨 이를 유효 출력에서 제외하고 coverage/무효점으로 기록했다.
3. **Scale/evaluation 분리:** reference 0만 scale fitting에 사용하고 refs 1–4만 지표에 포함했다. held-out ref의 GT는 추론·scale fitting에 사용하지 않았다.
4. **distortion-aware GT:** ETH3D 문서상 `_depth.7z` depth는 원본 distorted 이미지 좌표의 float32 depth이며 invalid는 +Inf다. 이를 공식 THIN_PRISM_FISHEYE 모델로 undistorted PINHOLE grid에 remap한 뒤 평가했다. [ETH3D depth/calibration 문서](https://eth3d.ethz.ch/documentation)
5. **주장 범위:** 결과는 5 scenes × 4 held-out references의 exploratory pilot으로 제한했다. 프레임 선택은 RGB-only COLMAP track support로 고정했다.

## 미완료 항목과 원인

공식 V-FUSE README는 baseline MVS input data를 “coming soon”으로 적고 GBiNet depth/confidence/camera/image 입력 레이아웃을 설명한다. 공개 V-FUSE checkpoint로 fusion 자체는 실행했지만, 저자 제공 GBiNet train-domain depth/confidence 예측 세트를 확보하지 못해 권고된 train-domain control은 완료되지 않았다. VGGT-Omega 예측으로 이 control을 대체하지 않았다. [V-FUSE 공식 구현 및 데이터 준비 안내](https://github.com/nburgdorfer/V-FUSE)

P3_001–003은 중간 진단 시도로 남겼다. 첫 구현은 delivery_area/min–max의 비양수 depth를 예외 처리해 scene을 중단했다. 양수 confidence 하한을 적용해도 모델이 유한한 음수 depth를 냄을 확인했고, 최종 P3_004는 유한 출력은 보존하되 nonpositive depth를 무효 prediction으로 처리해 전체 조건을 끝까지 실행했다.

## 재현 자료

- P0: checkpoint/source/data provenance, SHA-256, 환경 정보
- P1: frozen frame manifests, distorted→undistorted GT remap 및 audit
- P2: scene별 Omega raw cache와 metrics
- P3_004: 100 V-FUSE inference output 및 scene summary
- P4: 25 oracle-camera inference output 및 scale 기록
- P5_002: `heldout_metrics.csv`, `confidence_diagnostics.csv`, `risk_coverage.csv`, `summary.json`

현재 수치는 공식 ETH3D leaderboard 프로토콜과 동일한 전체 장면 point-cloud 평가가 아니라 선택 reference의 sparse depth-map 진단이다. confidence map 변환과 V-FUSE 모델의 입력-domain 차이도 결과 해석에 남는 제한이다.
