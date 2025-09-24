# 센서 데이터 생성 알고리즘 분석 회의록

## 개요
- **날짜**: 2025-09-04
- **주제**: LLaVA-Sensing 데이터 생성에서 센서 데이터 값 매핑 알고리즘 분석
- **분석 범위**: Simple vs Advanced 모드의 센서 값 생성 방식

## 주요 질의사항
1. **Advanced 모드의 작동 방식**: 키워드 사전을 통한 작동 여부
2. **Simple vs Advanced 차이점**: 둘 다 키워드 사전 사용 시 차이점
3. **센서 데이터 값 매핑 알고리즘**: 실제 센서 값 생성 메커니즘

## 분석 결과

### 1. Advanced 모드 작동 방식
- **키워드 사전 기반**: `corpus_based_generator.py`의 `SensorKeywords` 클래스 활용
- **키워드 분류**:
  - 온도: `temperature_hot` (뜨거운, 더운, 무더운...), `temperature_cold` (차가운, 추운, 시원한...)
  - 습도: `humidity_dry` (건조한, 메마른...), `humidity_wet` (습한, 촉촉한, 젖은...)
  - 바람: `wind_keywords` (바람, 미풍, 산들바람...)
  - 움직임: `movement_keywords` (걷다, 뛰다, 달리다...)
  - 감각: `sensory_keywords` (느끼다, 감각, 촉감...)

### 2. Simple vs Advanced 모드 차이점

#### Simple 모드 특징
- **단순한 키워드 매칭**: 기본적인 키워드 개수만 계산
- **직접적인 센서값 추론**: 키워드 개수에 따라 랜덤 값 생성
- **의존성 없음**: numpy, scikit-learn 등 라이브러리 불필요
- **파일**: `simple_novel_processor.py`

#### Advanced 모드 특징  
- **복잡한 분석 파이프라인**:
  - `CorpusAnalyzer`: TF-IDF, 코사인 유사도 등 고급 텍스트 분석
  - `SensorContextExtractor`: 문맥적 센서 정보 추출
  - `AdvancedSensorMapper`: 정교한 센서값 매핑
  - `LiteraryStyleClassifier`: 문학 스타일 분류
- **신뢰도 점수**: confidence 계산으로 매핑 품질 평가
- **균형잡힌 데이터셋**: 가중치 기반 샘플링으로 품질 높은 예제 선별
- **무거운 의존성**: numpy, pandas, nltk, scikit-learn 필요
- **파일**: `novel_corpus_processor.py`, `corpus_processor.py`

### 3. 센서 데이터 값 매핑 알고리즘

#### Simple 모드 - 규칙 기반 직접 매핑

**온도 알고리즘** (`simple_novel_processor.py:137-154`):
```python
if temp_hot > temp_cold:
    if temp_hot > 2:
        temperature = random.uniform(25, 35)  # 고온
    else:
        temperature = random.uniform(20, 28)  # 온화
elif temp_cold > 0:
    if temp_cold > 2:
        temperature = random.uniform(-5, 10)  # 극한 추위
    else:
        temperature = random.uniform(5, 18)   # 서늘함
else:
    temperature = random.uniform(10, 25)     # 중간 온도
```

**습도 알고리즘** (`simple_novel_processor.py:155-162`):
```python
if humidity_wet > humidity_dry:
    humidity = random.uniform(60, 90)    # 습함
elif humidity_dry > 0:
    humidity = random.uniform(20, 45)    # 건조
else:
    humidity = random.uniform(40, 70)    # 기본
```

**특징**:
- 키워드 개수 기반 단순 분류
- 고정 범위 내 균등분포 랜덤값 생성
- 빠른 처리 속도

#### Advanced 모드 - 가우시안 분포 기반 매핑

**온도 매핑** (`corpus_processor.py:300-352`):
```python
temp_mappings = {
    ('hot', 2): (30, 40, 5),      # (평균, 최대값, 표준편차)
    ('hot', 1): (25, 32, 3),
    ('cold', 2): (-5, 10, 8),
    ('cold', 1): (10, 20, 5),
    ('neutral', 0): (15, 25, 8)
}

temperature = np.random.normal(mean, std)  # 가우시안 분포
temperature = max(-20, min(max_val, temperature))  # 범위 제한
```

**습도 매핑** (`corpus_processor.py:308-363`):
```python
humidity_mappings = {
    ('humid', 2): (80, 100, 10),   # (평균, 최대값, 표준편차)
    ('humid', 1): (60, 85, 12),
    ('dry', 2): (10, 30, 8),
    ('dry', 1): (30, 50, 10),
    ('neutral', 0): (40, 70, 15)
}

humidity = np.random.normal(mean, std)
humidity = max(0, min(100, humidity))  # 0-100% 범위 제한
```

**바람 방향 매핑**:
```python
wind_direction_mappings = {
    'front': (0, 0.3),            # (기준각도, 노이즈)
    'back': (π, 0.3),
    'side': (π/2, 0.5),
    'unknown': (0, 2π)            # 균등 랜덤
}
```

**특징**:
- 강도별 세분화된 분류 (level, intensity)
- 가우시안 분포 기반 자연스러운 값 생성
- 통계적으로 더 현실적인 센서 데이터

### 4. IMU 데이터 생성

**움직임 기반 IMU 매핑**:
```python
movement_imu_mappings = {
    ('running', 3): {'acc_std': 3.0, 'gyro_std': 0.5},
    ('walking', 2): {'acc_std': 1.5, 'gyro_std': 0.2},
    ('climbing', 2): {'acc_std': 2.0, 'gyro_std': 0.3},
    ('standing', 1): {'acc_std': 0.5, 'gyro_std': 0.1},
    ('stationary', 0): {'acc_std': 0.2, 'gyro_std': 0.05}
}
```

## 결론

### 핵심 차이점 요약
| 구분 | Simple 모드 | Advanced 모드 |
|------|------------|---------------|
| **방식** | 키워드 개수 → 고정 범위 균등분포 | 강도별 분류 → 가우시안 분포 |
| **정교함** | 단순한 규칙 기반 | 통계적 분포 기반 |
| **의존성** | 기본 라이브러리만 | ML/과학 컴퓨팅 라이브러리 |
| **품질 평가** | 없음 | 신뢰도 점수 기반 필터링 |
| **데이터 현실성** | 기본적 | 통계적으로 더 현실적 |

### 권장사항
- **빠른 프로토타이핑**: Simple 모드 사용
- **고품질 데이터셋 생성**: Advanced 모드 사용
- **리소스 제약 환경**: Simple 모드 사용
- **연구/실험용**: Advanced 모드 사용

## 관련 파일
- `preprocess_with_args.py`: 메인 실행 스크립트
- `simple_novel_processor.py`: Simple 모드 구현
- `novel_corpus_processor.py`: Advanced 모드 메인 로직
- `corpus_processor.py`: Advanced 모드 센서 매핑 구현
- `corpus_based_generator.py`: 키워드 정의 및 분석 로직