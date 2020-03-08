import datetime
import random
import json
import os
import logging
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import TextDatasetConfig, DarkHistoryDataset
from model import ReformerGenConfig, ReformerGenModel

logging.basicConfig(level=logging.DEBUG, filename="0307.log")
logger = logging.getLogger(__name__)

class TrainerConfig:

    def __init__(self, **kwargs):
        self.save_dir = kwargs.pop("save_dir", "../model/")
        self.save_model_name = kwargs.pop("save_model_name", "trained_model.bin")
        self.save_config_name = kwargs.pop("save_config_name", "config.bin")

        self.trained_model_path = kwargs.pop("trained_model_path", None)

        self.logging_step = kwargs.pop("logging_step", 50)
        self.save_step = kwargs.pop("save_step", 10000)

        self.seed = kwargs.pop("seed", 17)

        self.n_gpu = kwargs.pop("n_gpu", 1)
        self.use_device_id = kwargs.pop("use_device_id", [0])

        self.epoch = kwargs.pop("epoch", 10)
        self.batch_size = kwargs.pop("batch_size", 8)
        if self.n_gpu > 1:
            self.batch_size *= self.n_gpu

        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.max_grad_norm = kwargs.pop("max_grad_norm", 1.0)


    @staticmethod
    def from_json(path:str):
        kwargs = json.load(open(path, 'r', encoding='utf-8', errors='ignore'))
        return TrainerConfig(**kwargs)

class Trainer:

    def __init__(self, config:TrainerConfig,
        model_config_path   = "../config/reformer_config.json",
        dataset_config_path = "../config/dataset_config.json"
    ):

        self.config = config

        self.set_seed(config.seed)
        self.device = torch.device(f"cuda:{self.config.use_device_id[0]}" if torch.cuda.is_available() else "cpu")

        dataset_config = TextDatasetConfig.from_json(dataset_config_path)
        self.dataset = DarkHistoryDataset(dataset_config)

        self.train_dataloader = DataLoader( self.dataset,
                                            batch_size=self.config.batch_size,
                                            shuffle=True,
                                            num_workers=8)

        logger.info("Load Dataset : Complete")

        model_config = ReformerGenConfig.from_json(model_config_path)
        model_config.vocab_size = self.dataset.get_vocab_size()

        if self.config.trained_model_path is not None:
            self.model = ReformerGenModel.from_pretrained(self.config.trained_model_path, model_config, self.device)
        else:
            self.model = ReformerGenModel(model_config)

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr = self.config.learning_rate
        )

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if self.config.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def generate(self, start_sent:str) -> str:
        # 文の続きを生成します。
        model_to_gen = self.model.module if hasattr(self.model, "module") else self.model

        sent_ids = [self.dataset.cls_id] + self.dataset.sent_to_ids(start_sent)
        sent_ids = torch.tensor(sent_ids).to(self.device)

        sent = model_to_gen.generate(sent_ids, self.dataset.sep_id)
        return ''.join(self.dataset.ids_to_sent(sent))

    def random_generate(self, sent_num=5) -> List[str]:
        generated_sents = []
        for i in range(sent_num):
            start_char = random.randrange(self.dataset.mask_id+1, self.dataset.get_vocab_size())
            start_char = self.dataset.ids_to_sent([start_char])[0]
            sent = self.generate(start_char)
            logger.info(f"generate:{sent}")
            generated_sents.append(sent)

        return generated_sents

    def save_model(self, step):
        save_dir = os.path.join(self.config.save_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, self.config.save_model_name))
        torch.save(self.config, os.path.join(save_dir, self.config.save_config_name))

        logger.info(f"Saving model checkpoint to {step}")

    def logging(self, step, loss):
        logger.info(f"step : {step}")
        logger.info(f"loss : {loss/self.config.logging_step}")

    def train(self):
        if self.config.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids = self.config.use_device_id)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        self.model.zero_grad()
        self.model.train()

        logger.info("---- START TRAINING ----")
        logger.info(f"---- {datetime.datetime.now()} ----")
        for i in range(self.config.epoch):
            logger.info(f"Start epoch:{i}")
            for batch in self.train_dataloader:
                try:
                    inputs = batch["input_ids"].to(self.device)
                    mask = batch["input_mask"].to(self.device)

                    loss = self.model(inputs, input_mask=mask, lm_labels=inputs)[0]

                    if self.config.n_gpu > 1:
                        loss = loss.mean()

                    loss.backward()
                    tr_loss += loss.item()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

                except Exception as e:
                    # あとでエラー処理書く
                    logger.exception(f"raise error at step {global_step}")
                    raise Exception

                if self.config.logging_step > 0 and global_step % self.config.logging_step == 0:
                    self.logging(global_step, tr_loss-logging_loss)
                    logging_loss = tr_loss

                if self.config.save_step > 0 and global_step % self.config.save_step == 0:
                    self.save_model(global_step)
                    self.model.eval()
                    self.random_generate()
                    self.model.train()


        self.save_model(global_step)
        logger.info("---- FINISH TRAINING ----")
        logger.info(f"---- {datetime.datetime.now()} ----")


if __name__ == "__main__":
    config  = TrainerConfig.from_json("../config/train_config.json")
    trainer = Trainer(config)
    trainer.train()
