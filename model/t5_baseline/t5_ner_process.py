import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup
from model.ner_metric import NERMtric

from flask import Flask

app = Flask(__name__)


class T5NERProcess(object):
    """
    T5 NER模型训练类
    """

    def __init__(self, args, t5_model, t5_tokenizer, accelerator, logger):
        self.args = args
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.accelerator = accelerator
        self.logger = logger
        self.ner_metric = NERMtric(args)

    def train(self, train_dataloader, dev_dataloader):
        """
        训练T5 NER模型
        :param train_dataloader:
        :param dev_dataloader:
        :return:
        """
        self.t5_model.train()

        # prepare optimizer and shedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.t5_model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        t_total = len(train_dataloader) * self.args.epoch_num
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_ratio),
                                                    num_training_steps=t_total)

        # Prepare everything with "accelerator", 会统一将模型、数据加载到对应到device
        self.t5_model, optimizer, train_dataloader, dev_dataloader = self.accelerator.prepare(
            self.t5_model, optimizer, train_dataloader, dev_dataloader
        )

        # 进行到多少batch
        total_batch = 0
        dev_best_score = float("-inf")
        # 上次验证集loss下降的batch数
        last_improve = 0
        # 是否很久没有效果提升
        no_improve_flag = False

        self.logger.info(f"Total Train Batch Num: {len(train_dataloader)}")
        for epoch in range(self.args.epoch_num):
            self.logger.info(f"Epoch [{epoch + 1}/{self.args.epoch_num}]")
            for step, batch_data in enumerate(train_dataloader):
                outputs = self.t5_model(**batch_data)
                loss = outputs.loss
                self.accelerator.backward(loss)
                # 更新参数
                optimizer.step()
                scheduler.step()
                # 清空梯度
                optimizer.zero_grad()
                self.logger.info(f"step: {step}, train loss: {loss.item()}")

                # 输出在验证集上的效果（每隔一定数目的batch）
                if total_batch % self.args.eval_batch_step == 0 and epoch > 5:
                    eval_metric = self.evaluate(dev_dataloader)
                    if eval_metric["f1"] > dev_best_score:
                        dev_best_score = eval_metric["f1"]
                        self.accelerator.save(
                            self.accelerator.unwrap_model(self.t5_model).state_dict(), self.args.model_save_path
                        )
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""
                    self.logger.info(
                        f'Iter: {total_batch}, Train Loss: {loss.item()}, eval_metric: {eval_metric["f1"]} {improve}'
                    )
                    self.t5_model.train()
                total_batch += 1
                if total_batch - last_improve > self.args.require_improvement_step:
                    self.logger.info("No optimization for a long time, auto-stopping...")
                    no_improve_flag = True
                    break
            if no_improve_flag:
                break

    def evaluate(self, dev_dataloader):
        """
        评估模型
        :param dev_dataloader:
        :return:
        """
        self.t5_model.eval()

        all_input_list = []
        all_pred_list = []
        all_label_list = []

        with torch.no_grad():
            for step, batch_data in enumerate(dev_dataloader):
                batch_generate_ids = self.accelerator.unwrap_model(self.t5_model).generate(
                    batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    max_length=self.args.max_target_len,
                    num_beams=self.args.beam_num
                )
                # 不同进程生成的长度不一致因此需要padding
                batch_generate_ids = self.accelerator.pad_across_processes(
                    batch_generate_ids, dim=1, pad_index=self.t5_tokenizer.pad_token_id
                )

                batch_label_ids = batch_data["labels"]
                # 将多个不同设备上的tensor padding到同一维度（当使用的动态padding时，不同设备最大长度会不一致）
                if not self.args.pad_to_max_length:
                    batch_label_ids = self.accelerator.pad_across_processes(
                        batch_label_ids, dim=1, pad_index=self.t5_tokenizer.pad_token_id
                    )

                # 聚合多个不同设备的tensor并将其拼接
                generate_ids_gathered = self.accelerator.gather(batch_generate_ids).cpu().clone().numpy()
                label_ids_gathered = self.accelerator.gather(batch_label_ids).cpu().clone().numpy()

                # 将id解码为对应的token
                if self.args.ignore_pad_token_for_loss:
                    # 将label中的-100替换为tokenizer中的pad_id，否则无法解码
                    # np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y
                    label_ids_gathered = np.where(
                        label_ids_gathered != -100, label_ids_gathered, self.t5_tokenizer.pad_token_id
                    )
                generate_tokens = self.t5_tokenizer.batch_decode(generate_ids_gathered, skip_special_tokens=True)
                label_tokens = self.t5_tokenizer.batch_decode(label_ids_gathered, skip_special_tokens=True)
                input_tokens = self.t5_tokenizer.batch_decode(
                    batch_data["input_ids"].cpu().clone().numpy(), skip_special_tokens=True
                )

                self.logger.info(f"generate_result: {generate_tokens[0]}, label: {label_tokens[0]}")
                # self.logger.info(f"generate_result: {generate_tokens}, label: {label_tokens}")

                # 将当前batch结果加入评测
                self.ner_metric.add_generate_batch(batch_pred_tokens=generate_tokens, batch_label_tokens=label_tokens)

                all_input_list.extend(input_tokens)
                all_pred_list.extend(generate_tokens)
                all_label_list.extend(label_tokens)
        # 计算评测指标
        self.logger.info(self.ner_metric.static_dict)
        eval_metric = self.ner_metric.compute_generate_metric()

        return eval_metric

    def test(self, test_dataloader):
        """
        测试模型
        :param test_dataloader:
        :return:
        """
        # 加载模型
        self.t5_model.load_state_dict(torch.load(self.args.model_save_path))
        self.t5_model.eval()
        self.t5_model, test_dataloader = self.accelerator.prepare(self.t5_model, test_dataloader)

        eval_metric = self.evaluate(test_dataloader)
        self.logger.info("Test Score: {0}".format(eval_metric["f1"]))
