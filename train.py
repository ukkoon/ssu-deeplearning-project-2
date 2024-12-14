import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

df = pd.read_csv('training_conversations.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

# 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터셋 클래스 정의
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
        
        # input_ids와 attention_mask를 텐서로 변환
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return item

# 데이터셋 객체 생성 시 tokenizer 인자를 포함
train_dataset = ConversationDataset(train_df, tokenizer)
test_dataset = ConversationDataset(test_df, tokenizer)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()