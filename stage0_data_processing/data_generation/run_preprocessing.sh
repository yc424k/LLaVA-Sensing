#!/bin/bash

# 데이터 전처리 실행 스크립트
echo "=== LLaVA-NeXT 데이터 전처리 시작 ==="

# 작업 디렉토리 설정
cd /home/yc424k/LLaVA-NeXT/data_generation

# 출력 디렉토리 생성
mkdir -p data/processed
mkdir -p data/logs

# 로그 파일 설정
LOG_FILE="data/logs/preprocessing_$(date +%Y%m%d_%H%M%S).log"

echo "로그 파일: $LOG_FILE"
echo "시작 시간: $(date)" | tee $LOG_FILE

# 1. 단순 코퍼스 처리 (의존성 없음)
echo "=== 1. 단순 코퍼스 처리 실행 ===" | tee -a $LOG_FILE
python3 simple_novel_processor.py 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    echo "✓ 단순 코퍼스 처리 완료" | tee -a $LOG_FILE
else
    echo "✗ 단순 코퍼스 처리 실패" | tee -a $LOG_FILE
    exit 1
fi

# 2. 고급 처리 (의존성 있는 경우)
if command -v numpy &> /dev/null; then
    echo "=== 2. 고급 코퍼스 처리 실행 ===" | tee -a $LOG_FILE
    python3 novel_corpus_processor.py 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        echo "✓ 고급 코퍼스 처리 완료" | tee -a $LOG_FILE
    else
        echo "✗ 고급 코퍼스 처리 실패 (의존성 확인 필요)" | tee -a $LOG_FILE
    fi
else
    echo "⚠ NumPy 없음 - 고급 처리 건너뜀" | tee -a $LOG_FILE
fi

# 3. 합성 데이터 생성 (선택사항)
echo "=== 3. 합성 데이터 생성 (선택사항) ===" | tee -a $LOG_FILE
echo "합성 데이터 생성을 원하면 다음 명령 실행:"
echo "python3 synthetic_dataset_generator.py"

# 결과 확인
echo "=== 전처리 결과 ===" | tee -a $LOG_FILE
echo "생성된 파일들:" | tee -a $LOG_FILE
find data -name "*.json" -type f | tee -a $LOG_FILE

echo "완료 시간: $(date)" | tee -a $LOG_FILE
echo "=== 데이터 전처리 완료 ==="