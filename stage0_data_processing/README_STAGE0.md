# Stage 0 – Data Processing

이 단계는 센서-문학 학습에 필요한 데이터를 준비하는 전처리 구간입니다. 기존 `data_generation/` 폴더의 모든 스크립트와 산출물이 `stage0_data_processing/` 아래로 옮겨졌습니다.

## 폴더 구조
```
stage0_data_processing/
├── data_generation/          # 데이터 생성·전처리 스크립트와 결과물
├── scripts/data/             # 실험 보조 스크립트 (예: 청크 병합)
└── test_hybrid_processor.py  # 하이브리드 처리기 테스트
```

## 주요 스크립트 사용법

### 1. 데이터셋 생성 파이프라인 실행
```bash
conda activate SensingLLaVA
cd /home/yc424k/LLaVA-Sensing

python stage0_data_processing/data_generation/dataset_pipeline.py \
  --max_files 202 --verbose --chunk_size 100
```
옵션은 `stage0_data_processing/data_generation/README.md` 참고.

### 2. 기존 청크 폴더 병합
```bash
python stage0_data_processing/scripts/data/merge_sensor_chunks.py \
  stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_test \
  stage0_data_processing/data_generation/data/processed/test_val_30k_each/travel_test \
  --output stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_travel_test.json
```

### 3. 하이브리드 처리기 동작 확인
```bash
python stage0_data_processing/test_hybrid_processor.py
```

## 산출물
- 모든 전처리 결과(JSON, 통계, 로그 등)는 `stage0_data_processing/data_generation/data/processed/` 아래에 저장됩니다.
- 학습에서 사용할 `SENSOR_DATA_PATH`는 이 경로를 가리키도록 설정하세요.

