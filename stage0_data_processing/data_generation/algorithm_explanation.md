# 데이터 전처리 알고리즘 분석

## 1. 키워드 기반 감각 분석 (Keyword-based Sensory Analysis)

### 온도 분류 알고리즘
```python
# 키워드 사전 정의
temp_keywords = {
    'hot': ['hot', 'warm', 'burning', 'blazing', 'scorching', 'sweltering', 'heated', 'sultry'],
    'cold': ['cold', 'cool', 'chilly', 'freezing', 'frozen', 'icy', 'frigid', 'bitter']
}

# 키워드 카운팅
temp_hot = sum(1 for word in temp_keywords['hot'] if word in text_lower)
temp_cold = sum(1 for word in temp_keywords['cold'] if word in text_lower)

# 규칙 기반 분류
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

**특징:**
- **키워드 매칭**: 문맥상 온도 관련 단어 검색
- **빈도 기반**: 키워드 등장 횟수로 강도 결정
- **규칙 기반 매핑**: 키워드 수 → 온도 범위 매핑
- **확률적 값**: 범위 내에서 랜덤 값 생성

### 습도 분류 알고리즘
```python
humidity_keywords = {
    'humid': ['humid', 'moist', 'damp', 'wet', 'soggy', 'steamy', 'muggy'],
    'dry': ['dry', 'arid', 'parched', 'dusty', 'drought']
}

# 경쟁 기반 분류
if humidity_wet > humidity_dry:
    humidity = random.uniform(60, 90)    # 습함
elif humidity_dry > 0:
    humidity = random.uniform(20, 45)    # 건조함
else:
    humidity = random.uniform(40, 70)    # 중간
```

## 2. 움직임 기반 IMU 데이터 생성

### 움직임 강도 분류
```python
movement_keywords = ['walk', 'run', 'move', 'step', 'stride', 'pace', 'march', 'wander']

# 키워드 빈도 → 움직임 강도 → IMU 표준편차
if movement_count > 2:
    acc_std = 2.0      # 격렬한 움직임
    gyro_std = 0.3
elif movement_count > 0:
    acc_std = 1.0      # 보통 움직임  
    gyro_std = 0.15
else:
    acc_std = 0.3      # 정적 상태
    gyro_std = 0.05

# 가우시안 노이즈로 IMU 값 생성
imu_data = [
    random.gauss(0, acc_std),      # 가속도계 X
    random.gauss(0, acc_std),      # 가속도계 Y
    9.8 + random.gauss(0, acc_std * 0.2),  # 가속도계 Z (중력 포함)
    random.gauss(0, gyro_std),     # 자이로스코프 X
    random.gauss(0, gyro_std),     # 자이로스코프 Y
    random.gauss(0, gyro_std * 0.5) # 자이로스코프 Z
]
```

## 3. 컨텍스트 추론 알고리즘

### 시간 분류 (Sequential Search)
```python
time_keywords = {
    'dawn': 'dawn', 'morning': 'morning', 'noon': 'noon',
    'afternoon': 'afternoon', 'evening': 'evening', 'night': 'night'
}

# 첫 번째 매칭되는 키워드 선택
detected_time = 'afternoon'  # 기본값
for keyword, time_label in time_keywords.items():
    if keyword in text_lower:
        detected_time = time_label
        break  # 첫 번째 매칭에서 중단
```

### 날씨 분류 (Priority-based Classification)
```python
# 우선순위 기반 분류 (if-elif 체인)
if any(word in text_lower for word in ['rain', 'raining']):
    weather = 'rain'          # 최우선
elif any(word in text_lower for word in ['snow', 'snowing']):
    weather = 'snow'          # 두 번째 우선순위
elif any(word in text_lower for word in ['cloud', 'cloudy', 'overcast']):
    weather = 'cloudy'        # 세 번째 우선순위
else:
    weather = 'clear'         # 기본값
```

### 시나리오 분류 (Environment Detection)
```python
# 환경 키워드 기반 분류
if any(word in text_lower for word in ['forest', 'tree', 'wood']):
    scenario = 'forest_exploration'
elif any(word in text_lower for word in ['mountain', 'hill', 'climb']):
    scenario = 'mountain_climbing'
elif any(word in text_lower for word in ['sea', 'ocean', 'beach', 'shore']):
    scenario = 'beach_walking'
elif any(word in text_lower for word in ['city', 'street', 'building']):
    scenario = 'city_walking'
else:
    scenario = 'general_walking'
```

## 4. 감각 점수 계산 (Sensory Score Calculation)

```python
# 전체 감각 키워드 카운팅
total_sensory = temp_hot + temp_cold + humidity_wet + humidity_dry + wind_count + movement_count

# 정규화된 감각 점수
sensory_score = total_sensory / word_count if word_count > 0 else 0

# 임계값 기반 필터링
if sensory_score < 0.01:
    continue  # 감각 내용이 부족한 단락 제외
```

## 5. 알고리즘의 특징과 한계

### 장점
1. **단순성**: 의존성 없이 빠른 처리
2. **해석 가능성**: 규칙이 명확함
3. **확장성**: 새로운 키워드 쉽게 추가 가능
4. **실시간성**: 빠른 분류 속도

### 한계
1. **문맥 무시**: 단어 주변 맥락 고려하지 않음
2. **동의어 처리**: 사전에 없는 유사 표현 놓침
3. **부정문 처리**: "not cold" 같은 부정 표현 오분류
4. **복합 표현**: "scorching cold" 같은 복합 표현 처리 어려움

## 6. 고급 버전에서 사용하는 알고리즘

### 정규표현식 패턴 매칭
```python
# 방향성 있는 바람 패턴
wind_patterns = {
    'front': [r'정면.*바람', r'앞.*바람', r'마주.*바람'],
    'back': [r'뒤.*바람', r'등.*바람', r'밀어.*바람'],
    'side': [r'옆.*바람', r'측면.*바람', r'스치.*바람']
}

# 패턴 매칭으로 방향 감지
for direction, patterns in wind_patterns.items():
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            wind_context['direction'] = direction
```

### TF-IDF 기반 문체 분류 (고급 버전)
```python
# 문체별 특징 벡터화
vectorizer = TfidfVectorizer()
style_vectors = vectorizer.fit_transform(style_samples)

# 코사인 유사도로 분류
similarity = cosine_similarity(passage_vector, style_vectors)
predicted_style = styles[similarity.argmax()]
```

### 센서 매핑 신뢰도 계산
```python
def calculate_mapping_confidence(context):
    confidence_factors = []
    
    # 키워드 밀도 기반 신뢰도
    if context['temperature']['keywords']:
        confidence_factors.append(min(1.0, len(context['temperature']['keywords']) * 0.3))
    
    # 평균 신뢰도 계산
    return np.mean(confidence_factors)
```

## 7. 실제 처리 흐름

1. **텍스트 전처리**: 소문자 변환, 토큰화
2. **키워드 매칭**: 각 카테고리별 키워드 검색
3. **빈도 계산**: 키워드 등장 횟수 집계
4. **규칙 적용**: 빈도 → 센서값 매핑 규칙 적용
5. **확률값 생성**: 범위 내 랜덤값 또는 가우시안 노이즈
6. **품질 평가**: 감각 점수로 데이터 품질 평가

이 알고리즘들은 간단하지만 효과적으로 문학 텍스트에서 환경 정보를 추출하여 현실적인 센서 데이터로 변환합니다.