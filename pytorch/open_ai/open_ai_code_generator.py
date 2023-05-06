pip install transformers==4.5.0
pip install --quiet pytorch-lightning==1.2.7


import json
import pandas as pd
import numpy as np
import torch

# dataset and dataloader for functions
from torch.utils.data import Dataset, DataLoader
# lightning for data class
import pytorch_lightning as pl
# leveraging the model checkpoints
from pytorch_lightning.callbacks import ModelCheckpoint
# we can visualize performance of model
from pytorch_lightning.loggers import TensorBoardLogger
# splitting the data
from sklearn.model_selection import train_test_split
# color formatting in ANSII code for output in terminal
from termcolor import colored
# wraps the paragraph into a single line or string
import textwrap
# installing multiple utilities
# including optimizer , tokenizer and generation module
from transformers import (
     AdamW,
     T5ForConditionalGeneration,
     T5TokenizerFast as T5Tokenizer
 )
# showing bars for processes in notebook
from tqdm.auto import tqdm
# seaborn for visualizing
import seaborn as sns
# procedural import to matplotlib
from pylab import rcParams
# graphs
import matplotlib.pyplot as plt
# rcParams for setting default values to all plots
from matplotlib import rc
pl.seed_everything(42)

with open('train-v2.0.json') as f:
    d = json.load(f)
    dataframe = pd.DataFrame.from_dict(d)

import json
# load data using Python JSON module
with open('train-v2.0.json','r') as f:
    data = json.loads(f.read())

df_nested_list = pd.json_normalize(data, record_path =['data'])

pd.json_normalize(df_nested_list, record_path =['paragraphs'])


a_json = json.loads(json_string)
print(a_json)

df = pd.read_csv('SQuAD_csv.csv')

data = df[['question','text']]

data.columns = ['text','code']


class CodeDataset(Dataset):
    def __init__(self, data:pd.DataFrame, tokenizer:T5Tokenizer, text_max_token_len: int = 100,code_max_token_len: int = 128):

        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.code_max_token_len = code_max_token_len

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index : int):

        data_row = self.data.iloc[index]

        text = data_row["text"]
        text_encoding = tokenizer(text, max_length = self.text_max_token_len, padding = "max_length", truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")

        code_encoding = tokenizer(data_row["code"], max_length = self.code_max_token_len, padding = "max_length", truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")
        labels = code_encoding["input_ids"]
        labels[labels ==0] = -100
        return dict(text = text,code = data_row["code"],text_input_ids=text_encoding["input_ids"].flatten(),text_attention_mask=text_encoding["attention_mask"].flatten(),labels=labels.flatten(),labels_attention_mask=code_encoding["attention_mask"].flatten())



class CodeDataModule(pl.LightningDataModule):
    def __init__(self,train_df: pd.DataFrame,test_df: pd.DataFrame,tokenizer: T5Tokenizer,batch_size: int = 8,text_max_token_len: int = 100,code_max_token_len: int = 128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.code_max_token_len = code_max_token_len
    def setup(self,stage=None):
        self.train_dataset = CodeDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.code_max_token_len
        )
        self.test_dataset = CodeDataset(
                self.test_df,
                self.tokenizer,
                self.text_max_token_len,
                self.code_max_token_len
        )
    def train_dataloader(self):
        return DataLoader(
        self.train_dataset,
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = 2
        )
    def val_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = 2
        )
    def test_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = 2
        )



MODEL_NAME = "t5-base"
tokenizer =T5Tokenizer.from_pretrained(MODEL_NAME)
N_EPOCHS = 20
BATCH_SIZE = 8
data_module = CodeDataModule(train_df, test_df , tokenizer ,batch_size=BATCH_SIZE)


class TextCodeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
        input_ids,
        attention_mask = attention_mask,
        labels = labels,
        decoder_attention_mask = decoder_attention_mask
        )
        return output.loss, output.logits
    def training_step(self,batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self(
        input_ids = input_ids, 
        attention_mask = attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels= labels
        )
        
        self.log("train_loss",loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self,batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self(
        input_ids = input_ids, 
        attention_mask = attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels= labels
        )
        
        self.log("val_loss",loss, prog_bar=True, logger=True)
        return loss
    def test_step(self,batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self(
        input_ids = input_ids, 
        attention_mask = attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels= labels
        )
        
        self.log("test_loss",loss, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)


## Create an instance of the model class
model = TextCodeModel()
##clear up some unused memory
import gc
gc.collect()
##Logging model training into Tensor Board
%load_ext tensorboard
%tensorboard --logdir ./lightning_logs
## saving model checkpoints in a directory
checkpoint_callback = ModelCheckpoint(
  dirpath = "checkpoints",
  filename = "best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor = "val_loss",
  mode= "min"
)
logger = TensorBoardLogger("lightning_logs",name="text-code")
trainer = pl.Trainer(
  logger= logger,
  checkpoint_callback= checkpoint_callback,
  max_epochs= N_EPOCHS,
  gpus=1,
  progress_bar_refresh_rate= 30   
)
############ Training the model 
trainer.fit(model, data_module)
## Loading the trained model from checkpoint
trained_model = TextCodeModel.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path
)
## Freezing the model
trained_model.freeze()


def text_to_code(text):
    text_encoding = tokenizer(
        text,
        max_length=100,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
        )
    
    generated_ids = trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length= 100,
        num_beams = 2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        for gen_id in generated_ids]
    return "".join(preds)