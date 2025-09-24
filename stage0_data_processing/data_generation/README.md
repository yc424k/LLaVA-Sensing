# LLaVA-NeXT 데이터 전처리 가이드

## 📋 개요

Novel 코퍼스(모더니즘 소설 + 여행기)를 활용하여 센서-문학 변환 모델을 위한 학습 데이터를 생성하는 시스템입니다.

## 📂 파일 구조

```
stage0_data_processing/data_generation/
├── README.md                     # 이 파일
├── preprocess_with_args.py       # 통합 실행 스크립트 (추천)
├── simple_novel_processor.py     # 단순 처리기 (의존성 없음)
├── novel_corpus_processor.py     # 고급 처리기 (의존성 필요)
├── split_dataset.py              # 데이터셋 분할 유틸리티
├── dataset_reader.py             # 청크 데이터 읽기 유틸리티
├── algorithm_explanation.md      # 알고리즘 상세 설명
└── data/
    ├── novel_dataset.json        # 원본 통합 데이터셋
    └── chunks/                   # 분할된 청크 파일들
```

## 🚀 빠른 시작

### 1. 기본 실행 (소규모 테스트)

```bash
cd /home/yc424k/LLaVA-Sensing/stage0_data_processing/data_generation

# 각 카테고리당 3개 파일로 테스트
python3 preprocess_with_args.py --verbose
```

### 2. 전체 데이터 처리 (403개 파일)

```bash
# 모든 파일 처리 (권장)
python3 preprocess_with_args.py --max_files 202 --verbose --chunk_size 100

# 백그라운드 실행 (시간이 오래 걸릴 경우)
python3 preprocess_with_args.py --max_files 202 --verbose --chunk_size 100 > processing.log 2>&1 &
```

## ⚙️ 실행 모드

### Simple 모드 (기본, 의존성 없음)
- **특징**: 빠른 처리, 외부 라이브러리 불필요
- **알고리즘**: 키워드 기반 매칭, 규칙 기반 분류
- **실행**: `--mode simple` (기본값)

### Advanced 모드 (의존성 필요)
- **특징**: 정교한 분석, 높은 정확도
- **알고리즘**: 정규표현식, TF-IDF, 신뢰도 계산
- **의존성**: `pip install numpy pandas nltk scikit-learn`
- **실행**: `--mode advanced`

## 📝 명령어 옵션

### 필수 옵션
```bash
--max_files N          # 각 카테고리당 최대 N개 파일 처리
--verbose              # 상세 진행상황 출력
```

### 선택 옵션
```bash
--mode simple/advanced         # 처리 모드 선택 (기본: simple)
--output_dir PATH             # 출력 디렉토리 (기본: data/processed)
--chunk_size N                # 청크당 예시 개수 (기본: 100)
--min_passage_length N        # 최소 단락 길이 (기본: 150)
--max_passage_length N        # 최대 단락 길이 (기본: 500)
--min_sensory_score F         # 최소 감각 점수 (기본: 0.01)
--auto_split true/false       # 자동 청크 분할 (기본: true)
```

## 🎯 실행 예제

### 1. 전체 파일 처리
```bash
# 기본 설정으로 전체 처리
python3 preprocess_with_args.py --max_files 202 --verbose

# 작은 청크로 세밀 분할
python3 preprocess_with_args.py --max_files 202 --chunk_size 50 --verbose

# 고급 모드로 전체 처리
python3 preprocess_with_args.py --mode advanced --max_files 202 --verbose
```

### 2. 맞춤 설정
```bash
# 단락 길이 조정
python3 preprocess_with_args.py --max_files 50 --min_passage_length 200 --max_passage_length 400

# 감각 점수 필터링 강화
python3 preprocess_with_args.py --max_files 100 --min_sensory_score 0.02

# 자동 분할 비활성화
python3 preprocess_with_args.py --max_files 10 --auto_split false
```

### 3. 개별 스크립트 실행
```bash
# 단순 처리기만 실행
python3 simple_novel_processor.py

# 분할만 별도 실행
python3 split_dataset.py

# 배치 스크립트 실행
./run_preprocessing.sh
```

## 📊 데이터 규모 예상

| 파일 수 | 예상 데이터 수 | 처리 시간 | 청크 파일 수 (100개/청크) |
|---------|----------------|-----------|---------------------------|
| 6개     | ~1,900개       | 10초      | ~19개                     |
| 50개    | ~15,000개      | 1-2분     | ~150개                    |
| 202개   | ~125,000개     | 10-30분   | ~1,250개                  |

## 📈 처리 결과 확인

### 1. 진행상황 모니터링
```bash
# 실시간 로그 확인
tail -f processing.log

# 생성된 파일 확인
ls -la data/
ls -la data/chunks/
```

### 2. 데이터셋 통계 확인
```bash
# 청크 데이터 읽기 테스트
python3 dataset_reader.py

# 통계 파일 확인
cat data/novel_dataset_stats.json
```

### 3. 청크 데이터 활용
```python
from dataset_reader import ChunkedDatasetReader

reader = ChunkedDatasetReader("data/chunks")
info = reader.get_info()
print(f"총 데이터: {info['total_examples']}개")

# 특정 청크 로드
chunk1 = reader.get_chunk(1)
print(f"첫 번째 청크: {len(chunk1)}개")

# 스타일별 검색
travel_data = reader.search_by_style('travel')
print(f"여행기 스타일: {len(travel_data)}개")
```

## 🔧 문제 해결

### 1. 의존성 오류
```bash
# Advanced 모드 의존성 설치
pip install numpy pandas nltk scikit-learn

# 또는 가상환경 사용
python3 -m venv preprocessing_env
source preprocessing_env/bin/activate
pip install numpy pandas nltk scikit-learn
```

### 2. 메모리 부족
```bash
# 더 작은 청크 크기 사용
python3 preprocess_with_args.py --max_files 202 --chunk_size 50

# 적은 파일 수로 분할 처리
python3 preprocess_with_args.py --max_files 50 --verbose
python3 preprocess_with_args.py --max_files 100 --verbose
```

### 3. 처리 시간 단축
```bash
# Simple 모드 사용 (빠름)
python3 preprocess_with_args.py --mode simple --max_files 202

# 더 큰 단락 길이로 데이터 수 감소
python3 preprocess_with_args.py --min_passage_length 300 --max_files 202
```

## 📋 데이터 형식

### 생성되는 JSON 구조
```json
{
  "id": "A001_The_Awakening_passage_001",
  "sensor_data": {
    "temperature": 23.5,
    "humidity": 65.2,
    "wind_direction": 1.57,
    "imu": [0.1, 0.05, 9.8, 0.01, 0.02, 0.1],
    "context": {
      "scenario": "city_walking",
      "time": "afternoon", 
      "weather": "clear"
    }
  },
  "target_paragraph": "실제 문학 텍스트 내용...",
  "metadata": {
    "source_file": "A001_The_Awakening.txt",
    "style_category": "modernist",
    "sensory_analysis": {...},
    "passage_index": 1
  }
}
```

## 📚 추가 자료

- **알고리즘 상세**: `algorithm_explanation.md` 참고
- **코퍼스 구조**: `/home/yc424k/LLaVA-NeXT/Novel/` 참고
  - `Modernist_Novel/`: 202개 모더니즘 소설
  - `Travel_Novel/`: 201개 여행기

## 🎯 권장 워크플로우

1. **소규모 테스트**: `python3 preprocess_with_args.py --max_files 5 --verbose`
2. **중간 규모**: `python3 preprocess_with_args.py --max_files 50 --verbose`
3. **전체 처리**: `python3 preprocess_with_args.py --max_files 202 --verbose --chunk_size 100`
4. **결과 확인**: `python3 dataset_reader.py`

---

## ❓ FAQ

**Q: 모든 403개 파일을 처리해야 하나요?**
A: 필요에 따라 조정하세요. 6개 파일로도 1,900개 데이터가 생성되므로, 초기 실험에는 50-100개 파일로도 충분할 수 있습니다.

**Q: Advanced 모드와 Simple 모드의 차이는?**
A: Simple은 빠르고 의존성이 없지만, Advanced는 더 정교한 분석을 제공합니다. 초기에는 Simple로 시작을 권장합니다.

**Q: 청크 파일이 너무 많아지면?**
A: `--chunk_size`를 늘려서 더 큰 청크를 만들거나, 처리할 파일 수를 줄이세요.

**Q: 처리가 중단되면?**
A: 이미 생성된 청크 파일들은 유지되므로, `dataset_reader.py`로 확인 후 필요시 이어서 처리하세요.
