import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import evaluate  # evaluate 라이브러리 사용

# 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ConversationDataset 클래스
class ConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        human_input = self.dataframe.iloc[idx]['human_input']
        bot_response = self.dataframe.iloc[idx]['bot_response']
        labels = 1 if self.dataframe.iloc[idx]['linguistic_acceptability'] == 'yes' else 0
        
        inputs = self.tokenizer(
            human_input,
            bot_response,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return item

# 검증 데이터 로드
val_df = pd.read_csv('validating_conversations.csv')
val_dataset = ConversationDataset(val_df, tokenizer)

# 평가 메트릭
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def evaluate_checkpoint(checkpoint_dir, val_dataset):
    # 모델 로드
    model = BertForSequenceClassification.from_pretrained(checkpoint_dir)

    # 평가 설정
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 평가
    eval_result = trainer.evaluate()
    
    return eval_result['eval_accuracy']

# 체크포인트 디렉터리 리스트
checkpoint_dirs = ['./results/checkpoint-500', './results/checkpoint-1000',
                   './results/checkpoint-1500', './results/checkpoint-2000',
                   './results/checkpoint-2151'
                   ]

accuracies = []
for checkpoint in checkpoint_dirs:
    accuracy = evaluate_checkpoint(checkpoint, val_dataset)
    accuracies.append(accuracy)
    
print(accuracies)