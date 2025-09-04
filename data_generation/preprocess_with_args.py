#!/usr/bin/env python3
"""
매개변수를 받는 데이터 전처리 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Novel corpus preprocessing for LLaVA-Sensing')
    
    # 입력/출력 경로
    parser.add_argument('--input_dir', default='/home/yc424k/LLaVA-Sensing/Novel',
                       help='Novel corpus input directory')
    parser.add_argument('--output_dir', default='data/processed',
                       help='Output directory for processed data')
    
    # 처리 옵션
    parser.add_argument('--max_files', type=int, default=5,
                       help='Maximum files per category to process')
    parser.add_argument('--min_passage_length', type=int, default=150,
                       help='Minimum passage length in characters')
    parser.add_argument('--max_passage_length', type=int, default=500,
                       help='Maximum passage length in characters')
    parser.add_argument('--min_sensory_score', type=float, default=0.01,
                       help='Minimum sensory content score')
    
    # 모드 선택
    parser.add_argument('--mode', choices=['simple', 'advanced'], default='simple',
                       help='Processing mode: simple (no deps) or advanced (with deps)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--auto_split', action='store_true', default=True,
                       help='Automatically split dataset into chunks (default: True)')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Number of examples per chunk when splitting')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Processing mode: {args.mode}")
        print(f"Max files per category: {args.max_files}")
    
    try:
        if args.mode == 'simple':
            from simple_novel_processor import SimpleNovelProcessor
            
            processor = SimpleNovelProcessor(args.input_dir)
            examples = processor.process_corpus(args.max_files)
            
            output_file = f"{args.output_dir}/dataset_{args.max_files}files.json"
            stats = processor.save_dataset(examples, output_file)
            
            if args.verbose:
                print(f"\n=== Processing Complete ===")
                print(f"Total examples: {stats['total_examples']}")
                print(f"Output saved to: {output_file}")
        
        elif args.mode == 'advanced':
            try:
                from novel_corpus_processor import NovelCorpusProcessor
                
                processor = NovelCorpusProcessor(args.input_dir)
                results = processor.process_novel_collection(
                    max_files_per_category=args.max_files,
                    min_confidence=0.4
                )
                
                dataset = processor.create_balanced_dataset(results, target_size=1000)
                stats = processor.save_dataset(dataset, args.output_dir)
                
                if args.verbose:
                    print(f"\n=== Advanced Processing Complete ===")
                    print(f"Total examples: {stats['total_examples']}")
                    print(f"Output saved to: {args.output_dir}")
            
            except ImportError as e:
                print(f"Error: Advanced mode requires additional dependencies: {e}")
                print("Try running: pip install numpy pandas nltk scikit-learn")
                sys.exit(1)
    
        # Auto-split dataset if requested
        if args.auto_split and args.mode == 'simple':
            try:
                from split_dataset import split_json_dataset
                
                if args.verbose:
                    print(f"\n=== Auto-splitting dataset ===")
                
                chunks_dir = f"{args.output_dir}/chunks"
                split_json_dataset(output_file, chunks_dir, args.chunk_size)
                
                if args.verbose:
                    print(f"Dataset split into chunks of {args.chunk_size} examples")
                    print(f"Chunks saved to: {chunks_dir}")
                    
            except ImportError:
                if args.verbose:
                    print("Warning: Could not import split_dataset module for auto-splitting")
            except Exception as split_error:
                print(f"Warning: Auto-split failed: {split_error}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()