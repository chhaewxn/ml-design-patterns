# **우리가 공부한 것 🙇**
## CHAPTER 8. 연결 패턴

> 한 문장: ML 디자인 패턴은 단독으로 쓰이는 것이 아니라, 서로 연결되어 조합으로 힘을 발휘한다. 패턴 간의 상호작용을 이해하고, 프로젝트 수명 주기와 사용 사례에 맞게 적절히 엮는 것이 실무의 핵심!

---

### 8.1 패턴 참조

**8.1.1 디자인 패턴**

- 이 책의 패턴들은 ML 워크플로 순서대로 정리되어 있음
    - 입력 표현 → 문제 표현 → 모델 학습 → 탄력성 → 재현성 → 책임 있는 AI
- 실무에서는 하나의 프로젝트에 여러 패턴이 동시에 적용됨
    - 예: Rebalancing(문제 표현) + Checkpoint(학습) + Continuous Model Evaluation(탄력성) 조합

---

**8.1.2 데이터 표현 패턴**

> 모델에 넣기 전에 입력을 어떻게 가공할지의 문제

- **Feature Hash**: 카테고리 Feature가 결정적이고 이식 가능한 문제라면 해시 함수를 적용. 버킷 수 조정으로 제어 가능
- **Embedding**: 밀집 벡터를 높은 카디널리티 Feature에 적용. 관련 항목을 가까이 배치하여 모델이 유사성을 학습
- **Feature Cross**: 개별 Feature만으로 학습하기 어려운 관계를 입력 간 조합으로 명시적으로 표현. 모델이 Feature 간 관계를 빠르게 학습 가능
- **Multimodal Input**: 이미지, 텍스트, 수치 등 여러 데이터 표현을 동시에 모델에 입력하는 방식

---

**8.1.3 학습 패턴에서 놓치기 쉬운 것들**

- **Useful Overfitting**: 물리 기반 모델이나 동적 시스템에 ML을 적용할 때, 의도적으로 Overfitting하여 근사 함수로 사용하는 경우. 일반적인 ML 원칙과 반대이므로 주의
- **Checkpoint**: 긴 시간 동안 수행되는 학습 작업이 중간에 고장으로 중단될 가능성이 있을 때, 모델 상태를 주기적으로 저장. 처음부터 다시 학습하는 것을 방지
- **Transfer Learning**: 정교한 모델을 처음부터 학습시키는 대신, 사전 학습된 모델의 가중치를 재사용하여 적은 데이터로도 높은 성능 달성

---

**8.1.4 탄력성 패턴**

> 배포 이후에 모델이 살아남는 방법

- **Stateless Serving Function**: 모델 입력에 필요한 Feature 변환을 학습과 서빙 양쪽에서 반드시 동일하게 유지. 학습/서빙 간 불일치(Training-Serving Skew)가 대표적인 장애 원인
- **Batch Serving**: 한 번에 다수의 요청을 처리하도록 설계. 분산 데이터 인프라를 활용하여 여러 인스턴스에서 비동기적으로 추론 수행
- **Continuous Model Evaluation**: 배포된 모델이 시간에 따라 성능이 저하되는지 지속적으로 모니터링. 데이터 Drift, 파이프라인 변경 등으로 모델이 더 이상 목적에 적합하지 않게 되는 경우를 감지
- **Two-Phase Prediction**: 엣지 환경이나 분산 환경에서 크고 복잡한 모델의 성능 유지가 필요한 경우, 간단한 모델과 정밀한 모델을 두 단계로 나눠 실행

---

**8.1.5 재현성 패턴**

> 같은 결과를 다시 만들 수 있는지

- **Transform**: 입력값을 모델에 사용하기 위한 특징 변환을 명시적으로 수집 및 저장. 학습과 서빙 과정에서 일관성을 유지
- **Repeatable Splitting**: 데이터 분할 시 랜덤 시드에만 의존하면 데이터 변경 시 재현 불가. 행들의 상관관계를 파악하여 열을 기준으로 분할하고, 동일 Algorithm을 학습/검증/테스트 데이터셋에 일관 적용
- **Bridged Schema**: 새로운 데이터로 인한 데이터 스키마 변경 시, 새 데이터뿐만 아니라 기존 데이터의 스키마에도 맞아야 이전 모델과의 호환성 유지
- **Feature Store**: 프로젝트 간 Feature를 공유할 수 있는 중앙화된 저장소. 여러 모델이나 모듈에서 동일 Feature를 사용할 때 중복 작업과 불일치를 방지

---

**8.1.6 책임 있는 AI**

> 기술적 패턴으로 해결해야 할 윤리적 문제

- **Heuristic Benchmark**: 모델의 복잡한 평가 지표로 인해 비즈니스 의사결정자가 성능을 이해하지 못하는 경우, 간단하고 이해하기 쉬운 직관적 지표와 함께 비교하여 제공
- **Explainable Prediction**: 규제, 규칙 준수 등을 위해 모델이 특정 예측을 내린 이유를 설명 가능하게 만들어야 함. 사용자 신뢰도를 높이기 위해 ML 시스템에 대한 설명 가능성 추가
- **Fairness Lens**: 학습 전 모든 사용자 데이터를 동등하게 취급하지 못하거나, 일부 정보가 부정적 영향을 줄 수 있는 경우, 학습 전 데이터셋의 편향을 식별하는 도구를 공정하게 적용

---

### 8.2 패턴 상호작용

> 패턴은 단독이 아닌 조합으로 사용

---

**8.2.1 Feature Store**

- Feature Store는 거의 모든 패턴의 **허브 역할**을 함
    - Embedding, Transform, Reframing, Bridge Schema 등에서 생성된 Feature를 저장/버전 관리/재사용
- Feature Store 없이 패턴을 개별 적용하면 팀 간 Feature 불일치, 중복 작업, Training-Serving Skew가 발생

---

**8.2.2 Transform + Stateless Serving Function + Feature Store**

> Training-Serving Skew를 구조적으로 방지하는 패턴 조합

- 학습 시 적용한 Feature 변환과 서빙 시 적용하는 변환이 달라지면 모델 성능이 급격히 저하
- **Transform 패턴**으로 변환 로직을 명시적으로 저장하고, **Stateless Serving Function**으로 서빙 시 동일하게 적용하며, **Feature Store**로 변환된 Feature의 버전을 관리하는 것이 정석 조합

---

**8.2.3 Reframing + Cascade + Rebalancing**

> 희귀 이벤트를 다루는 표준 조합

- 불균형 데이터를 Reframing으로 "정상 vs 이상" 분류로 변환 → Cascade로 1단계 분류 후 2단계 회귀 연결 → 1단계에서 Rebalancing 적용
- 이 조합에는 반드시 **Explainable Prediction 패턴을 함께 고려**해야 함. 여러 모델이 연쇄된 Cascade 구조에서는 최종 예측의 근거를 추적하기 어려워지기 때문

---

**8.2.4 Embedding**

> 다른 패턴과 가장 많이 연결되는 패턴

- **Feature Hash → Embedding**: 높은 Cardinality의 범주형 입력을 해시로 버킷화한 뒤 밀집 벡터로 변환하는 것이 표준 파이프라인
- **Transfer Learning**의 사전 학습된 모델 중간 계층의 출력값은 본질적으로 학습된 Feature Embedding
- **Neutral Class**를 추가하면 Embedding의 품질이 향상됨 (애매한 데이터가 분리되어 Embedding 공간이 깨끗해짐)

---

**8.2.5 Continuous Model Evaluation + Model Versioning + Workflow Pipeline**

- 배포된 모델은 시간이 지나면 성능이 저하됨 (Data Drift, 파이프라인 변경 등)
- **Continuous Model Evaluation**으로 성능 저하를 감지하고, **Model Versioning**으로 이전 버전과 비교하며, **Workflow Pipeline**으로 재학습 트리거를 자동화
- **Key-based Prediction** 패턴으로 예측 결과에 정답 데이터를 결합하면 Continuous Evaluation의 정확도가 높아짐

<aside>
💡 패턴을 개별적으로 이해하는 것도 중요하지만, 실무에서는 패턴 간의 조합이 성패를 가른다. 특히 Feature Store는 거의 모든 패턴의 허브 역할을 하므로 가장 먼저 구축을 고려해야 한다.
</aside>

---

### 8.3 ML 프로젝트 내의 패턴

**8.3.1 ML 수명 주기**

- ML 솔루션을 구축한다는 것
    1. 비즈니스 목표를 명확히 이해
    2. 궁극적으로 해당 목표에 도움이 되는 ML 모델을 프로덕션 환경에 포함시키는 프로세스

---

**발견 단계**

- **비즈니스 사용 사례 정의**
    - 모든 ML 프로젝트는 비즈니스 기회에 대한 철저한 이해와 현재 비즈니스 운영에 눈에 띄는 개선을 이뤄내는 방법으로 시작해야 함
    - 비즈니스 데이터를 이해하는 사람과 기술 팀의 협업이 성공 여부를 가를 만큼 중요
    - KPI(핵심 성과 지표)를 만들면 모든 사람이 공동적인 목표를 위해 노력하게 만들 수 있음
    - 기본 모델은 간단하더라도 설계 결정을 내리는 데 도움이 되며, 각 설계 선택으로 평가 지표상 성능 변화가 어떻게 이뤄지는지 이해하는 데 도움이 됨

- **데이터 탐색**
    - 비즈니스 심층 분석은 데이터 탐색 심층 분석과 함께 진행되어야 함
    - 나쁜 데이터는 나쁜 프로젝트를 만듦
    - EDA나 시각화를 통해 데이터의 특징을 파악하고 모델에 도움이 될지 확인 후 추가적으로 데이터 변환을 진행
    - '노이즈에 신호가 있는지' 확인하는 것이 매우 도움이 됨
    - 데이터의 타당성을 짧은 기술 스프린트로 확인하여 어떤 가공 단계가 유익한지 알 수 있음

---

**개발 단계**

- **데이터 파이프라인**: 데이터 입력을 모델에서 사용할 수 있도록 전처리하기 위해 필요
- **특징 가공**: 입력 데이터를 학습 목표와 가까워지게 함
    - 예: 버킷화, 형변환, 텍스트 토큰화, 형태소 분석, 범주형 특징 생성, 원-핫 인코딩, 입력 해싱, 특징 교차, 특징 임베딩 생성 등
- **모델 디자인 선택**: 문제에 맞는 다양한 디자인 패턴을 선택
- **ML 모델 구축**: 여러 번의 실험 후 데이터, 비즈니스 목표, KPI 등을 재검토. 새로운 조정이나 접근 방식은 발견 단계에서 설정된 평가 지표에 의해 측정됨
- **ML 모델 평가**: 분석 결과의 해석과 비즈니스 이해관계자 그룹에 전달하는 것이 중요 (숫자나 시각화 자료)

---

**배포 단계**

- **배포 계획 수립**
    - 모델이 비즈니스를 지원하는 프로덕션 환경에서 실행되어야 함
    - 배포 형태는 사용 사례나 조직에 따라 달라질 수 있음 (대화형 대시보드, 정적 노트북 파일, 재사용 가능한 라이브러리, 웹 서비스 엔드포인트 등)
    - 고려 사항: 재학습 관리, 인풋 관리, 학습 수행 관리, 모델 추론 관리, 지연 시간 문제 확인 등
    - 다양한 기술 컴포넌트 간의 통합이 필요하며, 레거시 시스템을 다루는 것부터 기존 복잡한 변경 제어나 생산 프로세스까지 고려

- **모델 운영 (MLOps)**
    - ML 모델 자동화, 모니터링, 테스트, 유지, 관리 측면을 모두 다룸
    - 자동화된 워크플로 파이프라인은 효율적인 워크플로와 반복 가능한 프로세스를 통해 향후 모델 개발을 개선하고 문제를 빠르게 해결 가능

- **CI/CD 프로세스 (지속적인 통합 및 배포)**
    - 코드 개발 내에서 안정성, 재현성, 속도, 보안, 버전 제어에 중점
    - 데이터 정리, 버전 관리, 데이터 파이프라인 조정 등에 CI/CD 원칙을 적용하면 많은 이점을 누릴 수 있음

- **모델 모니터링**
    - 모델 부실화에 따라 모델 보수 및 재학습이 필요할 수 있음
    - 시간이 지남에 따라 데이터의 분포가 변경되는 경우 (트렌드의 변화)
    - 데이터 드리프트로 인한 불균형 발생
    - 외부 소스 API 호출이 실패하거나 출력 형식의 변화로 생긴 오류가 있을 경우

<aside>
💡 ML 수명 주기는 발견 → 개발 → 배포의 선형 흐름이 아니라, 각 단계가 반복적으로 순환하는 과정이다. 배포 후 모니터링 결과가 다시 발견 단계로 피드백되어야 한다.
</aside>

---

**8.3.2 AI 준비 단계**

| 단계 | 특징 | 인프라 수준 |
| --- | --- | --- |
| **전술 단계 (수동 개발)** | 단기 프로젝트 중심의 작은 회사. 개념 증명/프로토타입에 국한. 데이터는 오프라인이나 격리된 저장소에 보관되며 수동 액세스. 자동화 도구 없음, 조직 내 공유 어려움 | 개발 전용 하드웨어 없음 |
| **전략 단계 (파이프라인 활용)** | 숙련된 팀과 전략적 파트너가 있는 회사. AI를 비즈니스 우선 순위와 일치시켜 핵심 가속 엔진으로 간주. 데이터는 웨어하우스에 저장, 중앙집중식 관리. 자동화된 ML 워크플로 파이프라인 운영, 성능 모니터링 트리거에 의해 실행 | 로깅, 모니터링, 알림이 있는 프로덕션 환경 |
| **혁신 단계 (완전 자동화)** | AI를 적극 사용하여 조직의 혁신을 촉진. 제품별 AI팀이 제품 팀에 포함되고 고급 분석 팀의 지원을 받음. 표준 도구, 라이브러리, 공통 패턴, 모범 사례가 쉽게 공유. 각 작업마다 CI/CD 통합 | GPU/TPU 같은 전문 ML 가속기 제공, 오케스트레이션된 실험 가능 |

---

### 8.4 사용 사례와 데이터 유형에 따른 일반적인 패턴

**8.4.1 NLU (Natural Language Understanding)**

> 텍스트와 언어의 의미를 이해하도록 기계를 학습시키는 데에 초점을 맞춘 AI 분야

- 세부 태스크: 텍스트 분류, 엔티티 추출, 질문 응답(QA), 음성 인식(STT), 텍스트 요약, 감성 분석
- 예시: Alexa(Amazon), Siri(Apple), Google Assistant(Google)
- 일반적으로 사용되는 디자인 패턴: Embedding, Feature Hashing, Neutral Class, Multimodal Input, Transfer Learning, Two-stage Prediction, Cascade, Window Inference

---

**8.4.2 CV (Computer Vision)**

> 이미지, 비디오, 아이콘, 픽셀 등의 시각적 입력을 이해하도록 기계를 학습시키는 AI 분야

- 세부 태스크: 이미지 분류, 비디오 동작 분석, 이미지 분할, 이미지 노이즈 제거
- 일반적으로 사용되는 디자인 패턴: Reframing, Cascade, Two-stage Prediction, Embedding, Feature Hashing, Multimodal Input, Transfer Learning, Multi-label, Neutral Class, Window Inference

---

**8.4.3 예측 분석**

> 과거 데이터를 사용하여 패턴을 찾고 미래에 특정 이벤트가 발생할 가능성을 결정

- 세부 태스크: 수요 예측(Demand Forecasting), 고객 이탈 예측(Churn Prediction)
- 일반적으로 사용되는 디자인 패턴: Feature Store, Feature Cross, Embedding, Ensemble, Transformation, Reframing, Multi-label, Neutral Class, Window Inference, Batch Serving

---

**8.4.4 RecSys (Recommendation System)**

> 과거 사용자들의 활동에서 유사한 특징을 포착하고 새롭게 주어진 사용자와 가장 관련성이 높은 항목을 추천

- 예시: 유튜브(시청 기록 기반 추천), 아마존(장바구니 기반 구매 추천)
- 일반적으로 사용되는 디자인 패턴: Embedding, Ensemble, Multi-label, Transfer Learning, Feature Store, Feature Hash, Reframing, Transformation, Window Inference, Two-stage Prediction, Neutral Class, Multimodal Input, Batch Serving

---

**8.4.5 Anomaly Detection**

> 데이터셋에서 비정상 동작이나 이상치를 찾는 기술

- 일반적으로 사용되는 디자인 패턴: Rebalancing, Feature Cross, Embedding, Ensemble, Two-stage Prediction, Transformation, Feature Store, Cascade, Neutral Class, Reframing

<aside>
💡 사용 사례마다 필요한 패턴 조합이 다르지만, Embedding, Feature Store, Transfer Learning은 거의 모든 도메인에서 공통적으로 등장한다. 자신의 프로젝트 도메인에 맞는 패턴 조합을 먼저 파악하는 것이 실무 적용의 첫걸음이다.
</aside>

---

## 질문모음.zip 🤔

**Q1. 모델 부실화(Model Drift) 현상에서의 리모델링 비용을 산정하는 방법은?**

A. 리모델링 비용은 여러 측면에서 산정할 수 있다: 신규 데이터 수집 및 정제(전처리 비용), 컴퓨팅 재학습(retraining) 비용, 모델 드리프트 파악을 위한 인적 자원 비용, 모델 검증 비용, 모델 성능 저하에 따른 기회비용, 배포 리스크 비용 등. 단기적으로 ROI를 계산하고, 장기적으로 브랜드 신뢰 등을 고려하여 리모델링을 진행한다.

---

## 단어집 📓

| **용어** | **정의** |
| --- | --- |
| Feature Hash | 카테고리형 Feature에 해시 함수를 적용하여 고정된 수의 버킷으로 매핑하는 기법 |
| Embedding | 높은 카디널리티의 Feature를 밀집 벡터로 변환하여 유사한 항목을 가까이 배치하는 표현 방식 |
| Feature Cross | 개별 Feature만으로 학습하기 어려운 관계를 입력 간 조합으로 명시적으로 표현하는 기법 |
| Multimodal Input | 이미지, 텍스트, 수치 등 여러 유형의 데이터를 동시에 모델에 입력하는 방식 |
| Transfer Learning | 사전 학습된 모델의 가중치를 재사용하여 적은 데이터로도 높은 성능을 달성하는 기법 |
| Checkpoint | 긴 학습 작업 중 모델 상태를 주기적으로 저장하여, 중단 시 처음부터 다시 학습하는 것을 방지하는 기법 |
| Useful Overfitting | 물리 기반 모델 등에서 의도적으로 Overfitting하여 근사 함수로 사용하는 패턴 |
| Training-Serving Skew | 학습 시와 서빙 시 Feature 변환이 달라 모델 성능이 저하되는 현상 |
| Feature Store | 프로젝트 간 Feature를 공유할 수 있는 중앙화된 저장소로, 중복 작업과 불일치를 방지 |
| Bridged Schema | 데이터 스키마 변경 시 새 데이터와 기존 데이터 모두에 호환되도록 유지하는 패턴 |
| Cascade | 여러 모델을 단계적으로 연결하여 1단계 분류 후 2단계 회귀 등으로 연쇄 실행하는 구조 |
| Heuristic Benchmark | 복잡한 모델 성능을 간단하고 직관적인 기준선과 비교하여 비즈니스 의사결정자에게 전달하는 패턴 |
| Explainable Prediction | 모델이 특정 예측을 내린 이유를 설명 가능하게 만드는 패턴 |
| Fairness Lens | 학습 전 데이터셋의 편향을 식별하고 공정하게 적용하는 도구/패턴 |
| MLOps | ML 모델의 자동화, 모니터링, 테스트, 유지, 관리를 포괄하는 운영 체계 |
| CI/CD | 지속적 통합(Continuous Integration) 및 지속적 배포(Continuous Deployment)를 통해 안정성, 재현성, 버전 관리에 중점을 두는 프로세스 |
| NLU (Natural Language Understanding) | 텍스트와 언어의 의미를 이해하도록 기계를 학습시키는 AI 분야 |
| Model Drift (모델 부실화) | 배포된 모델이 시간이 지나면서 데이터 분포 변화 등으로 성능이 저하되는 현상 |
