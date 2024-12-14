import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict

class TrainingDataProcessor:
    def __init__(self, dataset_path: str):
        """
        학습 데이터 처리를 위한 클래스 초기화
        
        Args:
            dataset_path (str): JSON 파일들이 있는 디렉토리 경로
        """
        self.dataset_path = dataset_path
        self.all_conversations = []

    def load_json_files(self, start_index: int = 1, end_index: int = 1000) -> List[Dict]:
        """
        지정된 범위의 JSON 파일들을 로드
        
        Args:
            start_index (int): 시작 파일 인덱스
            end_index (int): 종료 파일 인덱스
        
        Returns:
            List[Dict]: 로드된 모든 대화 데이터
        """
        processed_conversations = []
        
        for i in tqdm(range(start_index, end_index + 1), desc="Processing JSON files"):
            file_path = os.path.join(self.dataset_path, f"기술_과학_{i}.json")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                conversations = data['dataset']['conversations']
                # 각 대화 처리
                for conversation in conversations:
                    # print(conversations)
                    processed_conv = self._process_conversation(conversation)
                    if processed_conv:
                        processed_conversations.extend(processed_conv)
            
            except FileNotFoundError:
                print(f"파일 {file_path}을 찾을 수 없습니다.")
            except json.JSONDecodeError:
                print(f"파일 {file_path} JSON 디코딩 오류")
        
        return processed_conversations

    def _process_conversation(self, conversation: Dict) -> List[Dict]:
        """
        단일 대화 데이터 처리
        
        Args:
            conversation (Dict): 대화 데이터
        
        Returns:
            List[Dict]: 처리된 대화 데이터 리스트
        """
        processed_conversations = []
        
        # 메타데이터 추출
        topic = conversation['metadata'].get('topic', '기타')
        
        utterances = conversation.get('utterances', [])
        
        # 사용자와 봇의 대화 쌍 추출
        for i in range(0, len(utterances)-1, 2):
            human_utterance = utterances[i]
            bot_utterance = utterances[i+1]
            
            # 평가 지표 안전하게 추출
            evaluation = bot_utterance.get('utterance_evaluation', [])
            
            # 평가 지표가 존재하는 경우에만 처리
            linguistic_acceptability = 'unknown'
            consistency = 'unknown'
            
            if evaluation and isinstance(evaluation, list):
                first_eval = evaluation[0] if evaluation else {}
                linguistic_acceptability = first_eval.get('linguistic_acceptability', 'unknown')
                consistency = first_eval.get('consistency', 'unknown')
            
            processed_conv = {
                'topic': topic,
                'human_input': human_utterance.get('utterance_text', ''),
                'bot_response': bot_utterance.get('utterance_text', ''),
                'linguistic_acceptability': linguistic_acceptability,
                'consistency': consistency
            }
            
            processed_conversations.append(processed_conv)
        
        return processed_conversations

    def to_dataframe(self) -> pd.DataFrame:
        """
        처리된 대화 데이터를 DataFrame으로 변환
        
        Returns:
            pd.DataFrame: 대화 데이터 DataFrame
        """
        return pd.DataFrame(self.all_conversations)

    def save_processed_data(self, output_path: str = 'processed_conversations.csv'):
        """
        처리된 데이터를 CSV로 저장
        
        Args:
            output_path (str): 저장할 CSV 파일 경로
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"데이터가 {output_path}에 저장되었습니다.")

def main():
    # 데이터셋 경로 설정
    dataset_path = 'datasets/Validation'
    
    # 데이터 프로세서 초기화
    processor = TrainingDataProcessor(dataset_path)
    
    # JSON 파일 로드 및 처리 (예: 1~1000번 파일)
    processor.all_conversations = processor.load_json_files(1, 1000)
    
    # 처리된 데이터 CSV로 저장
    processor.save_processed_data('validating_conversations.csv')
    
    # 데이터 요약 출력
    df = processor.to_dataframe()
    print("\n데이터 요약:")
    print(f"총 대화 수: {len(df)}")
    print(f"주제별 분포:\n{df['topic'].value_counts()}")
    print(f"언어적 수용성 분포:\n{df['linguistic_acceptability'].value_counts()}")

if __name__ == "__main__":
    main()