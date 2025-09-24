# Stage 2 – Inference

최종 추론 단계 관련 스크립트와 배포용 엔트리포인트가 모여 있습니다. 기존 `scripts/inference/`와 `predict.py`가 이 폴더로 이동했습니다.

## 폴더 구조
```
stage2_inference/
├── scripts/
│   ├── run_sensor_inference.py
│   └── run_hybrid_inference.py
├── predict.py               # Cog 배포용
└── __init__.py
```

## 1. 로컬 추론
```bash
conda activate SensingLLaVA
cd /home/yc424k/LLaVA-Sensing

python stage2_inference/scripts/run_sensor_inference.py \
  --checkpoint checkpoints/sensor-literature/llama3-8b-modernist-travel-20250922 \
  --sensor_json stage0_data_processing/data_generation/data/processed/test_val_30k_each/modernist_travel_test.json \
  --image_file /home/yc424k/LLaVA-Sensing/src/ks-kyung-qnB45VQxo0o-unsplash.jpg \
  --prompt "Describe the current scene and environment in English."
```

### 옵션 요약
- `--sensor_json`: 전처리 단계에서 생성한 센서 JSON 파일
- `--image_file`: 선택 사항. 이미지가 없으면 센서 데이터만 사용
- `--conv_mode`, `--max_new_tokens`, `--temperature`: LLaVA 기본 옵션과 동일

## 2. Cog 배포 (옵션)

`cog.yaml`은 `stage2_inference/predict.py`를 엔트리포인트로 사용합니다. 배포가 필요할 때는 다음과 같이 실행하세요.
```bash
cog predict -i prompt="Describe the environment"
```

## 참고
- 스크립트 내부에서 `SensorDataProcessor`와 `EnvironmentalSensorEncoder`를 자동으로 활용합니다.
- 추론 단계에서 학습 시 사용한 센서 스케일링이 그대로 적용되어야 하므로, 전처리 단계에서 생성한 JSON 파일을 그대로 사용하세요.

