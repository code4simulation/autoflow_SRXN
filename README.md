# AutoFlow-SRXN: Automated Surface Reaction Workflow

**AutoFlow-SRXN**은 다양한 전구체(Precursor)와 기판(Substrate) 사이의 흡착 및 반응 구조를 고속으로 탐색하고 생성하기 위한 고도화된 자동화 프레임워크입니다.

---

## 🚀 주요 기능 (Key Features)

### 1. 지능형 전구체 파편화 및 중복 제거
- **그래프 기반 파편화**: RDKit 및 ASE를 활용하여 전구체 내 리간드를 자동으로 탐색합니다.
- **화학식 기반 그룹화**: 동일한 화학식을 가진 리간드(예: 3개의 H 리간드)를 자동으로 식별하고, 불필요한 반복 계산을 방지하여 탐색 효율을 극대화합니다.

### 2. 메커니즘 중심의 흡착 샘플링
- **물리 흡착 (Physisorption)**: 전구체 전체에 대한 구형 회전 및 높이별 배치 최적화.
- **해리형 화학 흡착 (Dissociative Chemisorption)**: 전구체 해리 경로와 표면 사이트 쌍(Dimer 등)을 정밀하게 매핑.
- **수소 교환 반응 (H-Exchange)**: 패시베이션된 표면에서 발생하는 리간드-수소 교환 구조 자동 생성.

### 3. 실리콘 표면 엔지니어링 (Si-Specific Utilities)
- **Si(100) 2x1 Reconstructed**: 버클링(Buckling) 정렬이 포함된 표준 재구성 표면 생성.
- **산화 및 패시베이션**: Greedy Max-Min 거리 알고리즘을 통한 균일한 산화막(Oxygen Bridge) 및 수소 패시베이션 처리.

### 4. 정밀 자가 진단 (Atomic-Level Diagnostics)
- **`verbose=True` 모드**: 라이브러리 내부에서 리간드 분석 및 스테릭 충돌 현황을 실시간 보고.
- **원자 수준 로그**: `[Overlap] Si(12) - C(45) clash (dist: 1.15 A)`와 같이 충돌 원자의 인덱스와 거리를 상세히 출력하여 디버깅 용이성 확보.

---

## 🏗️ 시스템 아키텍처 (Architecture)

### Core Libraries
- `example_dipas/ads_workflow_mgr.py`: 범용 흡착 워크플로우 매니저 클래스 (`AdsorptionWorkflowManager`).
- `example_dipas/surface_utils.py`: 기하 구조 기반 사이트 식별 및 범용 패시베이션 라이브러리.

### Specialty Modules
- `example_dipas/si_surface_utils.py`: 실리콘(Si) 표면 특화 재구성 및 산화/수화 유틸리티.

---

## 🛠️ 시작하기 (Quick Start)

DIPAS 전구체를 활용한 통합 연구 예제를 실행하려면 `example_dipas` 디렉토리에서 다음 명령을 수행하십시오.

```bash
cd example_dipas
python run_dipas_study.py
```

이 명령은 다음 4가지 표준 표면을 자동 생성하고 흡착 시뮬레이션을 수행합니다:
1. `Clean 2x1 Reconstructed`
2. `H-Passivated`
3. `Oxidized (50%)`
4. `Oxidized + H-Passivated`

---

## 📝 라이선스 및 연락처
본 프로젝트는 고효율 표면 반응 탐색을 위해 설계되었습니다. 문의 사항이 있으시면 담당 연구원에게 연락해 주시기 바랍니다.
