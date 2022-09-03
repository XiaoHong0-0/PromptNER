import os
import sys

# 引入包路径
sys.path.append("../MyPromptNER")

import transformers
import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator

from bean.arg_bean import T5ArgBean
from util.arg_parse import CustomArgParser
from util.log_util import LogUtil

from torch.nn.parallel import DataParallel

from model.t5_baseline.t5_ner_dataloader import T5NERDataLoader
from model.t5_baseline.t5_ner_process import T5NERProcess


class T5NERController(object):
    """
    T5 NER 模型整体框架
    """

    def __init__(self, args):
        self.args = args
        self.args.t5_config = AutoConfig.from_pretrained(self.args.pretrain_model_path)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_model_path)
        self.t5_ner_model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrain_model_path)

        self.accelerator = Accelerator()

        self.logger = LogUtil.logger

        self.t5_ner_dataloader = T5NERDataLoader(
            self.args, self.t5_ner_model, self.t5_tokenizer, self.accelerator, self.logger
        )
        self.t5_ner_process = T5NERProcess(
            self.args, self.t5_ner_model, self.t5_tokenizer, self.accelerator, self.logger
        )

    def train(self):
        """
        训练模型
        :return:
        """
        # 加载数据，返回DataLoader
        self.logger.info(self.accelerator.state)
        self.logger.info("Loading data ...")
        train_dataloader = self.t5_ner_dataloader.load_data(
            self.args.train_data_path, self.args.per_device_train_batch_size, is_train=True
        )
        dev_dataloader = self.t5_ner_dataloader.load_data(
            self.args.dev_data_path, self.args.per_device_train_batch_size, is_train=False
        )

        # 初始化模型前固定随机种子，保证每次运行结果一致
        transformers.set_seed(self.args.seed)

        # 训练模型
        self.logger.info("Training model ...")
        self.t5_ner_process.train(train_dataloader, dev_dataloader)
        self.logger.info("Finished Training model !!!")

    def test(self):
        """
        测试模型
        :return:
        """
        # 加载数据
        self.logger.info("Loading data ...")
        test_dataloader = self.t5_ner_dataloader.load_data(
            self.args.test_data_path, self.args.per_device_train_batch_size, is_train=False
        )
        self.logger.info("Finished loading data !!!")

        # 固定种子，保证每次运行结果一致
        transformers.set_seed(self.args.seed)

        # 测试模型
        self.logger.info("Testing model ...")
        self.t5_ner_process.test(test_dataloader)
        self.logger.info("Finished Testing model !!!")


if __name__ == '__main__':
    # 解析命令行参数
    args = CustomArgParser(T5ArgBean).parse_args_into_dataclass()
    # 传递参数给T5模型
    t5_ner_controller = T5NERController(args)
    # 模型训练
    if args.do_train:
        t5_ner_controller.train()
    # 模型测试
    if args.do_predict:
        t5_ner_controller.test()


