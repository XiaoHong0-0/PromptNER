import datasets
import random
import torch

from transformers import DataCollatorForSeq2Seq

from util.text_util import TextUtil


class T5NERDataLoader(object):
    """
    T5 NER模型数据加载类
    """

    def __init__(self, args, t5_model, t5_tokenizer, accelerator, logger):
        self.args = args
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.accelerator = accelerator
        self.logger = logger

    def load_data(self, data_path, batch_size, is_train=False):
        """
        加载数据
        :param data_path:
        :param batch_size:
        :param is_train:
        :return:
        """

        def tokenize_batch_func(batch_items):
            """
            处理批量训练数据
            :param batch_items:
            :return:
            """
            inputs = [TextUtil.truecase_sentence(token_list) for token_list in batch_items["token_list"]]
            outputs = [TextUtil.truecase_sentence(token_list) for token_list in batch_items["entity_list"]]

            # self.logger.info(outputs)

            if self.args.prompt:
                # 构造输入数据
                retrieval_inputs = [TextUtil.truecase_sentence(retrieval_list[0]['token_list']) for retrieval_list in
                                    batch_items['retrieval_list']]
                retrieval_inputs_label = [TextUtil.truecase_sentence(retrieval_list[0]['entity_list']) for
                                          retrieval_list in
                                          batch_items['retrieval_list']]
                #
                prompt_inputs = [retrieval_inputs[index] + ['</s>'] + retrieval_inputs_label[index] for index in
                                 range(len(retrieval_inputs))]
                #
                inputs = [inputs[index] + ['</s>'] + prompt_inputs[index] for index in range(len(inputs))]
                # inverse prompt
                # inputs = [prompt_inputs[index] + ['</s>'] + inputs[index] for index in range(len(inputs))]

                # if is_train:
                # 使用注解作为prompt
                #     inputs = [inputs[index] + ['</s>'] + outputs[index] + ['</s>'] + prompt_inputs[index] for index in range(len(inputs))]

                # print("retrieval_inputs: ", retrieval_inputs[0])
                # print("retrieval_inputs_label: ", retrieval_inputs_label[0])
                # print("prompt_inputs: ", prompt_inputs[0])
                # self.logger.info(f"inputs: {inputs[0]}")
                # print("inputs: ", inputs[0])

            # 构造输入数据
            model_inputs = self.t5_tokenizer(
                inputs,
                max_length=self.args.max_input_len,
                padding="max_length" if self.args.pad_to_max_length else False,
                truncation=True,
                # 已经预处理好为word list
                is_split_into_words=True
            )

            # 切分输出句子
            with self.t5_tokenizer.as_target_tokenizer():
                labels = self.t5_tokenizer(
                    outputs,
                    max_length=self.args.max_target_len,
                    padding="max_length" if self.args.pad_to_max_length else False,
                    truncation=True,
                    # 已经预处理好为word list
                    is_split_into_words=True
                )

            # self.logger.info(labels)

            # 将padding对应的token替换为-100，则后续loss将忽略对应的padding token
            if self.args.pad_to_max_length and self.args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(label_id if label_id != self.t5_tokenizer.pad_token_id else -100) for label_id in label_ids]
                    for label_ids in labels["input_ids"]
                ]

            model_inputs['labels'] = labels['input_ids']

            return model_inputs

        # 加载原始数据
        ner_dataset = datasets.load_dataset("json", data_files=data_path, split="train")

        # 打印部分数据供观察
        for index in random.sample(range(len(ner_dataset)), 3):
            self.logger.info(f"Sample {index} of the dataset: {ner_dataset[index]}\n")

        with self.accelerator.main_process_first():
            # 切分token，同时构造golden输出
            ner_dataset = ner_dataset.map(tokenize_batch_func, batched=True, batch_size=256,
                                          remove_columns=ner_dataset.column_names,
                                          num_proc=self.args.dataloader_proc_num)

        data_collator = DataCollatorForSeq2Seq(
            self.t5_tokenizer,
            model=self.t5_model,
            label_pad_token_id=-100 if self.args.ignore_pad_token_for_loss else self.t5_tokenizer.pad_token_id,
        )

        dataloader = torch.utils.data.DataLoader(
            ner_dataset, shuffle=is_train, collate_fn=data_collator, batch_size=batch_size
        )

        return dataloader
