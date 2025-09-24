# LLaVA-Sensing Code Map

간단히 논문 작성이나 코드 리뷰 시에 센서 융합 관련 파일들을 빠르게 찾을 수 있도록 정리했습니다. 기본 LLaVA 코드와 대비되는 연구 추가분 위주로 분류했습니다.

## 1. 데이터 전처리 & 유틸
- `stage1_training/train/modules/sensor_preprocessing.py`
  - 물리 센서 값을 정규화하고 텐서로 변환.
- `stage0_data_processing/data_generation/`
  - 센서-문학 데이터셋 생성 파이프라인과 전처리 산출물.
- `stage0_data_processing/scripts/data/merge_sensor_chunks.py`
  - 테스트/검증용 센서 청크 병합 스크립트.

## 2. 모델 구성 요소
- `llava/model/multimodal_encoder/environmental_sensor_encoder.py`
  - 센서 임베딩 및 크로스모달 어텐션.
- `llava/model/llava_arch.py`
  - LLaVA 메타 구조. 센서 인코더 초기화 로직 확인.
- `llava/model/language_model/llava_llama.py`
  - 언어 모델 진입점. `sensor_data` 토큰 삽입 및 이미지/텍스트 결합.

## 3. 학습 파이프라인
- `stage1_training/train/modules/train_sensor_literature.py`
  - 센서-문학 파인튜닝 스크립트.
- `stage1_training/train/finetune_sensor_literature*.sh`
  - 실험 실행용 셸 스크립트.
- `stage1_training/train/modules/sensor_literature_dataset.py`
  - 센서-문학 데이터셋 및 데이터 콜레이터.

## 4. 추론 & 데모 스크립트
- `stage2_inference/scripts/run_sensor_inference.py`
  - 센서 단독 혹은 이미지+센서 입력 지원.
- `stage2_inference/scripts/run_hybrid_inference.py`
  - 스트리밍 출력이 포함된 데모 스크립트.
- `stage2_inference/predict.py`
  - Cog 배포용 추론 엔트리포인트.
- `playground/`
  - 데이터 점검, 실험 보조 스크립트.

## 5. 체크포인트 & 리소스
- `checkpoints/sensor-literature/`
  - 연구용 파인튜닝 결과.
- `src/`
  - 논문 예시용 이미지 등 보조 자료.

## 6. 문서화 & 참고 자료
- `docs/`
  - 본 파일과 `training_and_inference.md` 등 연구 문서.

## 활용 팁
- 센서 관련 신규 코드는 `sensor_`, `environmental_` 등의 접두어를 사용.
- LLaVA 기본 구현과 비교 시, 위 목록의 파일만 우선 살펴보면 연구 기여분을 빠르게 파악 가능.
- 보다 세부적으로 정리하려면 이 문서를 업데이트하면서 실험별 노트를 추가하세요.
