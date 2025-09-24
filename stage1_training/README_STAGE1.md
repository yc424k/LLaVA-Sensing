# Stage 1 – Training

파이프라인의 학습 단계에 필요한 셸 스크립트와 설정이 위치한 구간입니다. 기존 `scripts/train/` 콘텐츠가 `stage1_training/train/`으로 이동했습니다.

## 폴더 구조
```
stage1_training/
├── train/
│   ├── finetune_sensor_literature.sh
│   ├── finetune_sensor_literature_llama3_8b.sh
│   ├── README.md, *.yaml, *.json …
│   └── zero*.json (DeepSpeed 설정)
└── __init__.py
```

## 기본 사용법
1. 환경 변수 설정
   ```bash
   conda activate SensingLLaVA
   cd /home/yc424k/LLaVA-Sensing
   export SENSOR_DATA_PATH="stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_travel_test.json"
   export OUTPUT_DIR="checkpoints/sensor-literature/llama3-8b-modernist-travel-$(date +%Y%m%d)"
   export NUM_GPUS=1
   ```

2. 학습 스크립트 실행
   ```bash
   bash stage1_training/train/finetune_sensor_literature_llama3_8b.sh
   ```
   추가 인자를 전달하려면 `--extra_args` 사용.

3. 진행 상황 모니터링
   ```bash
   bash stage1_training/train/finetune_sensor_literature_llama3_8b.sh | tee "$OUTPUT_DIR/train.log"
   ```

## 참고
- DeepSpeed 설정 파일은 그대로 `stage1_training/train/zero*.json`에 보관되어 있습니다.
- 파인튜닝 로직은 `stage1_training/train/modules/train_sensor_literature.py`에서 수행되며, 센서 인코더(`stage1_training/train/modules/environmental_sensor_encoder.py`) 사용 여부는 스크립트 인자에서 제어합니다.
