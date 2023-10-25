from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import os

class ModelTrainer:
    def __init__(self, config, param):
        self.config = config
        self.param = param
    
    def train(self):
        device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_ckpt'])
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config['model_ckpt']).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        dataset_samsum = load_from_disk(self.config['data_path'])

        trainer_args = TrainingArguments(
            output_dir=self.config['root_dir'],
            num_train_epochs = self.param['num_train_epochs'],
            warmup_steps = self.param['warmup_steps'],
            per_device_train_batch_size = self.param['per_device_train_batch_size'],
            per_device_eval_batch_size = self.param['per_device_eval_batch_size'],
            weight_decay = self.param['weight_decay'],
            logging_steps = self.param['logging_steps'],
            evaluation_strategy = self.param['evaluation_strategy'],
            eval_steps = self.param['eval_steps'],
            save_steps = self.param['save_steps'],
            gradient_accumulation_steps = self.param['gradient_accumulation_steps']
        )

        trainer = Trainer(
            model = model_pegasus,
            args = trainer_args,
            tokenizer = tokenizer,
            data_collator = seq2seq_data_collator,
            train_dataset = dataset_samsum['train'],
            eval_dataset = dataset_samsum['validation']
        )

        trainer.train()

        model_pegasus.save_pretrained(os.path.join(self.config['root_dir'],"pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config['root_dir'],"tokenizer"))
