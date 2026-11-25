# 현재까지 실험 요약

## 저장 위치와 구성

실험 코드 `VFUSE-20261124-003` 전체를 `results/VFUSE-20261124-003/`로 이동했다. 각 단계는 고유 ID 폴더이며 그 안에 코드, 결과, 로그를 분리했다.

```text
results/VFUSE-20261124-003/
├── P0_20261124-001/  # 코드·체크포인트·ETH3D 데이터 확보 및 provenance
├── P1_20261124-001/  # RGB-only frame 선택, GT distortion remap
├── P1_20261124-002/  # remap optical-center 수치 검산
├── P2_20261124-001/  # VGGT-Omega scene inference
├── P3_20261124-001/  # V-FUSE 초기 실행, 실패/부분 결과 보존
├── P3_20261124-002/  # confidence floor 재시도, 부분 결과 보존
├── P3_20261124-003/  # 음수 depth 진단
├── P3_20261124-004/  # 최종 100회 V-FUSE confidence ablation
├── P4_20261124-001/  # GT camera oracle 조건 25회
├── P5_20261124-002/  # 최종 held-out 평가와 confidence 진단
└── P6_20261124-001/  # paired color PLY export
```

## 완료한 작업

- ETH3D office, pipes, courtyard, delivery_area, kicker 다섯 장면을 준비했다.
- GT를 열기 전에 RGB/COLMAP track만으로 장면당 5개 view를 고정했다.
- undistorted PINHOLE 입력으로 VGGT-Omega 추론을 장면별 1회 실행했다.
- 원본 distorted float laser depth를 undistorted PINHOLE 좌표로 remap했다.
- V-FUSE confidence 입력 C1 constant, C2 min–max, C3 Q05–Q95, C4 empirical rank/CDF를 비교했다.
- reference 0으로 scale만 맞추고 refs 1–4, 총 20개 held-out frame에서 평가했다.
- GT PINHOLE 카메라/poses를 넣은 oracle-camera 비교도 별도로 실행했다.
- inference output SHA-256과 metrics를 대조했다. P3 최종 100개, P4 25개 output 모두 파일/해시 확인을 통과했다.
- held-out raw Omega와 V-FUSE C1–C4를 같은 predicted-camera 좌표계의 컬러 PLY로 변환했다. view별 100개, 4-view scene aggregate 25개, 별도 GT-camera P4 20개를 생성했다. 5개 장면 전체와 delivery_area 확대 PNG 미리보기도 만들었다.

## 주요 결과

20 held-out frame 평균 AbsRel은 Omega raw **0.0132**, V-FUSE C1 **0.0160**, C2 **0.1274**, C3 **0.0714**, C4 **0.0520**, C3 oracle-camera **0.0720**이다. 이 subset에서는 V-FUSE confidence 변환이나 oracle cameras가 depth 지표를 개선하지 않았다. C2–C4의 confidence 순위는 pixel error와 음의 Spearman 연관을 보였지만, 이는 확률 보정 결과가 아니며 depth 정확도 향상과도 구분된다.

P3 결과 전체에 유한한 0 이하 fused depth 22개가 있었고 모두 delivery_area에 나왔다. 해당 점을 예측 무효로 처리했으며 이번 sparse GT support와 겹치는 invalid point는 없어 held-out coverage는 전부 1.0이었다.

## 제한·미완료

- 이 결과는 ETH3D 5 scene × 4 held-out views의 exploratory pilot이다. ETH3D 전체 benchmark/point-cloud 성능 결론은 아니다.
- 저자 제공 GBiNet train-domain depth/confidence inputs를 이용한 control은 확보하지 못했다. 공식 V-FUSE 안내가 baseline MVS input data를 제공 예정으로 표시한다.
- remap center 검산은 왜곡식의 optical center 투영만 확인한다. image-wide inverse remap 정확도 전체를 입증하지 않는다.

자세한 표와 방법은 [`P5_20261124-002/report.md`](P5_20261124-002/report.md), 결과 CSV/JSON은 같은 stage의 `results/`에 있다.
