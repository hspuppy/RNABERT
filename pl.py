# based on pytorch lightning framework
"""
GPP IS score regression task.
Data: sample/IS_all.csv 4500

采用RNABert项目预训练好的Embedding，加上一个回归Head做回归任务。
"""
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import coloredlogs
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dataload import convert, kmer, make_dict, mask
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset
from utils.bert import BertModel, BertPreTrainingHeads, get_config

logger = logging.getLogger(__file__)
coloredlogs.install()


class RNASeqMLMDataset(Dataset):
    """RNA seq for MLM dataset, that is, only sequence"""
    def __init__(self, dataframe):
        self.data = dataframe
        self.maskrate = 0.2
        self.mag = 1
        self.max_length = 25

    def __getitem__(self, index):
        seq = self.data.iloc[index]['seq']
        # convert seq to ids
        gapped_seq = seq.upper().replace('T', 'U')
        seq = gapped_seq.replace('-', '')
        assert set(seq) <= set(['A', 'T', 'G', 'C', 'U'])
        assert len(seq) < 440
        k = 1
        kmer_seqs = kmer([seq], k)  # 1-mer序列，单碱基拆开，['A','U',...]
        masked_seq, _ = mask(kmer_seqs, rate = self.maskrate, mag = self.mag)  # 某些位置MASK某些位置随机换，low_seq=kmer_seqs
        kmer_dict = make_dict(k)  # MASK+碱基 -> num
        # swap_kmer_dict = {v: k for k, v in kmer_dict.items()}  # num -> base
        masked_seq_ids = np.array(convert(masked_seq, kmer_dict, self.max_length))[0]  # char convert to number
        seq_ids = np.array(convert(kmer_seqs, kmer_dict, self.max_length))[0]

        return {
            'seq': seq_ids, 
            'masked_seq': masked_seq_ids
        }

    def __len__(self):
        return len(self.data)


class RNABertForMLM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()
        # logger.info(f'hyperparameters: \n{self.hparams}')
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)  # TODO: 改成MLM任务专用头

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, attention_show_flg=False):
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)

        # 预训练分类头有三种pred, pred_ss, seq_relationship，第三个0/1分类的弃用了
        prediction_scores, prediction_scores_ss, seq_relationship_score = self.cls(encoded_layers, pooled_output)
        return prediction_scores, prediction_scores_ss, encoded_layers

    def on_train_end(self):
        pass

    def train_dataloader(self):
        data = pd.read_csv('./sample/IS_train.csv', usecols=['seq','score'])
        ds = RNASeqMLMDataset(data)
        return DataLoader(ds, 8, num_workers=4, shuffle=True)  # TODO: batch size from config

    def val_dataloader(self):
        data = pd.read_csv('./sample/IS_valid.csv', usecols=['seq','score'])
        ds = RNASeqMLMDataset(data)
        return DataLoader(ds, 8, num_workers=4, shuffle=False)  # TODO: batch size from config

    def test_dataloader(self):
        data = pd.read_csv('./sample/IS_test.csv', usecols=['seq','score'])
        ds = RNASeqMLMDataset(data)
        return DataLoader(ds, 8, num_workers=4, shuffle=False)  # TODO: batch size from config

    def training_step(self, batch, batch_idx):
        seq, masked_seq = batch['seq'], batch['masked_seq']
        pred_scores, _, _ = self.forward(masked_seq)  # scores for masked position
        # compare seq and predicted seq with socres, using cross entropy loss
        criterion = torch.nn.CrossEntropyLoss()
        mask = masked_seq - seq != 0  # 不一样的位置是masked
        same_base_mask = torch.bernoulli(torch.ones(mask.shape)*0.05)  # 按0.05的概率产生1，和0，噪音？ 
        same_base_mask = same_base_mask.to('cuda')  # TODO fix this
        mask = mask + same_base_mask  # add noise? 某些位置True + 1会变成2
        index = torch.nonzero(mask).split(1, dim=1)  # torch.nonzero->每行2d，是一个非0元素的坐标
        pred_scores = torch.squeeze(pred_scores[index])  # squeeze压缩掉所有维度为1的 1266,1,6 -> 1266,6，每行6个数表示类别概率
        seq_cat = torch.squeeze(seq[index])  # 1266，每行一个数表示类别
        loss = criterion(pred_scores, seq_cat)  # 比较每个mask位置真实分类和预测分类概率分布之差   
        # _, preds = torch.max(pred_scores, 1)  # 返回每行最大值和index, 每个mask具体预测是什么
        # train_acc = torch.sum(preds == seq_cat).item() / (len(seq_cat)*1.0)
        # self.log('acc', train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, masked_seq = batch['seq'], batch['masked_seq']
        pred_scores, _, _ = self.forward(masked_seq)  # scores for masked position
        # compare seq and predicted seq with socres, using cross entropy loss
        criterion = torch.nn.CrossEntropyLoss()
        mask = masked_seq - seq != 0  # 不一样的位置是masked
        index = torch.nonzero(mask).split(1, dim=1)  # torch.nonzero->每行2d，是一个非0元素的坐标
        pred_scores = torch.squeeze(pred_scores[index])  # squeeze压缩掉所有维度为1的 1266,1,6 -> 1266,6，每行6个数表示类别概率
        seq_cat = torch.squeeze(seq[index])  # 1266，每行一个数表示类别
        loss = criterion(pred_scores, seq_cat)  # 比较每个mask位置真实分类和预测分类概率分布之差   
        _, preds = torch.max(pred_scores, 1)  # 返回每行最大值和index, 每个mask具体预测是什么
        val_acc = torch.sum(preds == seq_cat).item() / (len(seq_cat)*1.0)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # ids, mask, y = batch['ids'], batch['mask'], batch['targets']
        # _, logits = self.forward(ids, mask)
        # loss = F.cross_entropy(logits, y)
        # y_hat = torch.argmax(logits, dim=1)
        # test_acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        # kappa = cohen_kappa(y_hat, y, N_CLASS)
        # self.log_dict({
        #     'test_loss': loss,
        #     'test_acc': test_acc,
        #     'test_kappa': kappa
        # })
        # logger.info(test_acc)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)  # TODO: config
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=3e-4, type=float)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--early_stop', type=str, default='val_loss')
        # parser.add_argument('--val_check_interval', type=int, default=10)
        return parser


def train_mlm():
    # loading dataframes and tokenize
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser = RNABertForMLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    config = get_config(file_path = "./RNA_bert_config.json") 
    config.hidden_size = config.num_attention_heads * config.multiple

    model = RNABertForMLM(config)
    # with pretrained parameters
    states = torch.load('./pretrained/bert_mul_2.pth')
    new_states = OrderedDict([(k.replace('module.', ''), v) for (k,v) in states.items()])
    model.load_state_dict(new_states)
    # print(model)
    # trainer = Trainer.from_argparse_args(args, gpus=1, max_epochs=20) 
    early_stopping = EarlyStopping('val_loss')
    trainer = Trainer.from_argparse_args(
        args, 
        callbacks=[early_stopping], 
        # precision=16,
        gpus=1, 
        max_epochs=100)
    # trainer = Trainer.from_argparse_args(args)
    trainer.fit(model) 


def test():
    # model = BertClassifier.load_from_checkpoint('./saved_models/distilbert/model.pth')
    # model.eval()
    # print(model)
    # trainer = Trainer(gpus=1)
    # result = trainer.test(model)
    # print(result)
    pass


if __name__ == '__main__':
    # process_data()
    train_mlm()
    # test()

