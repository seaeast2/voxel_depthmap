# VGGT-Omega + V-FUSE 후속 실험 (VFUSE-20261124-003)

기존 최종 보고서 후속 권고를 stage별로 실행했다. 실행 코드는 `P0`–`P6`에 보관하고, 실패한 시도는 재사용/덮어쓰기 없이 별도 P3 stage에 남겼다.

현재까지의 요약은 [`summary.md`](summary.md), 상세 평가 보고서는 [`P5_20261124-002/report.md`](P5_20261124-002/report.md), PLY 안내는 [`P6_20261124-001/README.md`](P6_20261124-001/README.md)에 있다.


## Stage index

- `P0_20261124-001`: 모델 체크포인트·고정 소스 snapshot 및 ETH3D 추가 데이터 확보. Omega checkpoint / V-FUSE GBiNet checkpoint와 SHA-256은 `P0_20261124-001/results/`.
- `P1_20261124-001`: RGB-only view 선택, distorted laser GT의 undistorted PINHOLE remap. `P1_20261124-002`는 remap 중심점 산식의 수치 검산 범위를 명확히 기록.
- `P2_20261124-001`: GT를 열기 전에 5 scene Omega 추론 완료.
- `P3_20261124-001`–`003`: V-FUSE의 음수 depth 출력/양수 confidence 처리 진단 및 실패 로그 보존.
- `P3_20261124-004`: 전체 5 scene × 4 confidence 조건 × 5 reference 비교, 총 100회 V-FUSE inference.
- `P4_20261124-001`: ETH3D GT intrinsics/poses를 쓰는 oracle camera 진단. reference 0 GT로 scale을 맞추고 refs 1–4를 평가.
- `P5_20261124-001`: 첫 집계 결과. `P5_20261124-002`: 최종 결과, 장면별 표와 confidence rank 진단 포함.
- `P6_20261124-001`: held-out Omega/V-FUSE map을 paired color PLY 145개와 PNG 미리보기로 변환.

## 결과

최종 결과 및 제한은 [`P5_20261124-002/report.md`](P5_20261124-002/report.md)에 있다. 핵심 표는 `P5_20261124-002/results/heldout_metrics.csv`, confidence 진단은 `confidence_diagnostics.csv`, risk-coverage는 `risk_coverage.csv`다. 비교는 ETH3D five-scene exploratory pilot이며 benchmark 전체 성능 주장이 아니다.

## 실행 환경

GPU inference는 host RTX 5060에서 sandbox 밖으로 실행했다. 실험 코드는 run 003에 고정 복사했으며, Python 환경은 용량/재설치 회피를 위해 이전 run의 Omega venv를 재사용했다: `../../experiments/vggt_omega_vfuse/VFUSE-20261124-002/P0_20261124-001/.venvs/omega` (Python 3.11.18, PyTorch 2.15.0+cu132). P0 README와 각 metrics JSON에 실제 source/checkpoint 경로 및 해시가 있다.

## 실행 blocker

V-FUSE 공식 저장소는 baseline MVS input data가 “coming soon”이라고 표기하고, GBiNet 입력 포맷만 안내한다. 공개 pretrained V-FUSE checkpoint는 실행했지만, 저자 제공 train-domain GBiNet depth/confidence inputs를 이용한 독립 control은 이 run에서 재현할 수 없어 미완료로 명시했다. Omega 입력을 그 control로 간주하지 않았다.
