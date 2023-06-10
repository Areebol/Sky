<!--
 * @Descripttion: 阅读代码以及相关的挑檐
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-09 17:45:46
-->

## 基本情况
- [模型]6种模型权重 7B-13B(包括instruction, pretraining)
- [训练框架]Deepspeed Zero-1 + Gradient Checkpointing
  - [DeepSpeed](https://zhuanlan.zhihu.com/p/343570325)是微软推出的大规模模型分布式训练的工具，主要实现了ZeRO并行训练算法
  - [GradientCheck](https://zhuanlan.zhihu.com/p/448395808)是用于显存优化，用时间换空间，降低训练用的显存
- [训练成本]2 * 8 * A100 80G gpu

## LLama参数下载
[13b](https://huggingface.co/decapoda-research/llama-13b-hf/tree/main)
[7b](https://huggingface.co/decapoda-research/llama-7b-hf/tree/main)

## 文件结构
```bash
.
├── LICENSE.md
├── Readme.md
├── apply_delta.py # 转化权重脚本
├── conf # 模型参数配置
│   └── llama # llama模型的参数
│       ├── wiki # wiki模型
│       │   ├── llama_7b_flan_v1_4_ds.yaml
│       │   ├── llama_7b_flan_v1_4_reclor_eval_prompt.yaml
│       │   ├── vicuna_7b_reclor_eval_prompt.yaml
│       │   └── vicuna_7b_reclor_eval_prompt_v1_1.yaml
│       └── zh # 中文模型
│           ├── belle_7b_c3_d_eval_prompt_v1_0_test.yaml
│           ├── belle_7b_c3_m_eval_prompt_v1_0_test.yaml
│           ├── belle_7b_logiqav2_eval_prompt_v1_0_test.yaml
│           ├── linly_7b_c3_d_eval_prompt_v1_0_test.yaml
│           ├── linly_7b_c3_m_eval_prompt_v1_0_test.yaml
│           ├── linly_7b_logiqav2_eval_prompt_v1_0_test.yaml
│           ├── llama_13b_zh_instruct_v1_0_ds.yaml
│           ├── llama_7b_zh_c3_eval_prompt_v1_0_test.yaml
│           ├── llama_7b_zh_c3_m_eval_prompt_v1_0_test.yaml
│           ├── llama_7b_zh_instruct_c3_eval_prompt_v1_0_test.yaml
│           ├── llama_7b_zh_instruct_c3_m_eval_prompt_v1_0_test.yaml
│           ├── llama_7b_zh_instruct_coig_sft_v1_0_ds.yaml
│           ├── llama_7b_zh_instruct_v1_0_ds.yaml
│           ├── llama_7b_zh_logiqav2_eval_prompt_v1_0.yaml
│           └── llama_7b_zh_logiqav2_eval_prompt_v1_0_test.yaml
├── data # 数据
│   ├── __init__.py
│   ├── collators
│   │   ├── __init__.py
│   │   ├── flan.py
│   │   ├── misc.py
│   │   └── zh_instruct.py
│   ├── data_utils.py
│   ├── flan_combine.py
│   ├── flan_sample.py
│   ├── flan_shuffle.py
│   ├── preprocessor
│   │   ├── __init__.py
│   │   └── mmlu_merge.py
│   ├── readers.py
│   └── reclor_prompt.py
├── general_util # 通用模块
│   ├── __init__.py
│   ├── average_meter.py
│   ├── dist_utils.py
│   ├── evaluator.py
│   ├── fsdp_utils.py
│   ├── lightseq_utils.py
│   ├── logger.py
│   ├── metrics.py
│   ├── mixin.py
│   ├── tensorboard_helper.py
│   ├── tokenization_utils.py
│   ├── torch_fsdp_utils.py
│   └── training_utils.py
├── make_delta.py # 转化导出权重脚本
├── models # 模型
│   ├── __init__.py
│   ├── llama.py
│   ├── roberta.py
│   └── t5.py
├── modules # 模块？
│   ├── __init__.py
│   ├── layers.py
│   ├── logits_processor.py
│   └── trie.py
├── panda_logo.PNG
├── post_processors # 后处理模块？
│   ├── bleu.py
│   ├── dist_mixin.py
│   ├── erica_post_processor.py
│   ├── reclor.py
│   └── retrieval.py
├── requirements.txt
├── seed_multi_run.sh
├── trainer_base_ds_mul.py
├── trainer_base_ds_mul_aws.py
├── trainer_base_ds_mul_tb.py
├── trainer_base_ds_v1.py
├── trainer_base_fsdp_mul.py
├── trainer_base_fsdp_v4.py
├── trainer_torch_fsdp.py
├── trainer_torch_fsdp_wandb.py
└── unify_format.ipynb
```

## 训练脚本
```bash
.
├── trainer_base_ds_mul.py
├── trainer_base_ds_mul_aws.py
├── trainer_base_ds_mul_tb.py
├── trainer_base_ds_v1.py
├── trainer_base_fsdp_mul.py
├── trainer_base_fsdp_v4.py
├── trainer_torch_fsdp.py
├── trainer_torch_fsdp_wandb.py
```
根据分布式训练的方法不同分成
- ds 未知的分布式方法
- fsdp 使用pytorch的FSDP方法

## 阅读`trainer_base_ds_v1.py`

```python
# 引用的包结构
import glob
import json
import logging
import os
import sys
from typing import Dict, Union


import deepspeed
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, PreTrainedTokenizer)

from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, unwrap_model, set_seed, note_best_checkpoint
```


```python
# convert sys.arg
if __name__ == "__main__":
    # 空列表，存储转化后的参数
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    # 读取命令行传递参数
    for arg in sys.argv:
        if arg.startswith("--"):
            # 去掉--加入到参数列表
            hydra_formatted_args.append(arg[len("--"):])
        else:
            # 直接加入参数
            hydra_formatted_args.append(arg)
    # 参数转化为hydra格式
    sys.argv = hydra_formatted_args
    main()
```
**训练模型 command**
```bash
# 使用Hydra配置参数
# 用deepspeed框架来运行脚本 trainer_base_ds_mul.py
# 使用gpu -> --include localhost:0,1,2,3,4,5,6,7
# 传入的配置信息: -cn config_name -cp config_path
HYDRA_FULL_ERROR=1 deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mul.py -cp conf/llama/zh -cn <file name of yaml config> 
```

**配置文件，device设置**
```python
# 指定配置文件路径，配置文件名称
@hydra.main(config_path="conf", config_name="config")
# 接收Hydra配置
def main(cfg: DictConfig):
    # 确定设备类型，分布式训练或cpu训练(？)
    if cfg.local_rank == -1 or cfg.no_cuda:
        # 配置cuda
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    # 单进程单卡训练
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        # dist.init_process_group(backend='nccl')
        # deepspeed分布式初始化方法
        deepspeed.init_distributed()
        # 单进程单gpu
        cfg.n_gpu = 1
        # 总的进程数量
        cfg.world_size = dist.get_world_size()
    cfg.device = device
```

**日志设置**
```python
    # 日志记录器
    global logger
    # 配置输出文件以及gpu rank
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    # 输出配置信息
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)

    # Set seed 保证实验可重复
    set_seed(cfg)
```

**模型加载**
```python
# 加载模型 Load pre-trained model and tokenizer
    # 确保第一个进程会下载模型和词汇表，其他等待
    if cfg.local_rank not in [-1, 0]:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # 加载预训练模型参数
    if cfg.pretrain:
        pretrain_state_dict = torch.load(cfg.pretrain, map_location='cpu')
    else:
        pretrain_state_dict = None

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    # 创建模型
    model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=pretrain_state_dict)

    if cfg.local_rank == 0:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # if cfg.local_rank == -1:  # For FullyShardedDDP, place the model on cpu first.
    #     model.to(cfg.device)

    # 打印日志
    # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
    if cfg.local_rank in [-1, 0] and cfg.do_train:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))
```

**模型训练**
```python
    # Training
    if cfg.do_train:
        # TODO: Add option for continuously training from checkpoint.
        #  The operation should be introduced in ``train`` method since both the state dict
        #  of schedule and optimizer (and scaler, if any) should be loaded.
        # If output files already exists, assume to continue training from latest checkpoint (unless overwrite_output_dir is set)
        continue_from_global_step = 0  # If set to 0, start training from the beginning
        # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
        #     if len(checkpoints) > 0:
        #         checkpoint = checkpoints[-1]
        #         logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
        #         continue_from_global_step = int(checkpoint.split('-')[-1])
        #         model = model_class.from_pretrained(checkpoint)
        #         model.to(args.device)

        # 加载数据集，特征
        train_dataset, features = load_and_cache_examples(cfg, tokenizer, _split="train")
        # train
        global_step, tr_loss = train(cfg, train_dataset, features, model, tokenizer, continue_from_global_step)
        # 输出到日志
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
```

**模型测试**
```python
# Test
    # 存放结果
    results = {}
    # 输出日志
    if cfg.do_eval and cfg.local_rank in [-1, 0]:
        checkpoints = [cfg.output_dir]
        if cfg.save_best:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
            checkpoints = [cfg.prediction_cfg.best_checkpoint]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.eval_sub_path:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model.bin", recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info(" the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            split = "dev"

            model = hydra.utils.call(cfg.model, checkpoint)
            model.to(device)

            if cfg.test_file:
                prefix = f'test' + (f'-{prefix}' if prefix != "" else "")
                split = "test"

            result = evaluate(cfg, model, tokenizer, prefix=prefix, _split=split)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results
```

### Train
```python
def train(cfg, train_dataset, features, model, tokenizer, continue_from_global_step=0):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        _dir_splits = cfg.output_dir.split('/')
        _log_dir = '/'.join([_dir_splits[0], 'runs'] + _dir_splits[1:])
        tb_writer = SummaryWriter(log_dir=_log_dir)
        tb_helper = hydra.utils.instantiate(cfg.summary_helper,
                                            writer=tb_writer) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_writer = None
        tb_helper = None

    cfg.train_batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
    train_sampler = RandomSampler(train_dataset) if cfg.local_rank == -1 else DistributedSampler(train_dataset)
    train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=cfg.train_batch_size,
                                  collate_fn=train_collator, num_workers=cfg.num_workers, pin_memory=True,
                                  prefetch_factor=cfg.prefetch_factor)

    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (len(train_dataloader) // cfg.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // cfg.gradient_accumulation_steps * cfg.num_train_epochs

    num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

    if "extended_vocab" in cfg and cfg.extended_vocab:
        logger.info(f"Extended extra vocab size: {cfg.extended_vocab}")
        model.resize_token_embeddings(model.config.vocab_size + cfg.extended_vocab)

    ds_config = cfg.ds_cfg
    ds_config.scheduler.params.total_num_steps = t_total
    ds_config.scheduler.params.warmup_num_steps = num_warmup_steps
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    # FIXME: Not supported in CPUAdam? Open an issue.
    # no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': cfg.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]

    model, optimizer, _, scheduler = deepspeed.initialize(model=model,
                                                          model_parameters=model.parameters(),
                                                          config=ds_config)
    logger.info(optimizer.optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                cfg.train_batch_size * cfg.gradient_accumulation_steps * (dist.get_world_size() if cfg.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
        if cfg.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(epoch_iterator):
            # If training is continued from a checkpoint, fast forward
            # to the state of that checkpoint.
            if global_step < continue_from_global_step:
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    global_step += 1
                continue

            model.train()
            batch = batch_to_device(batch, cfg.device)

            loss, outputs = forward_step(model, batch)
            loss /= cfg.gradient_accumulation_steps

            tr_loss += loss
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                global_step += 1

                # Log metrics
                if cfg.local_rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / cfg.logging_steps, global_step)
                    logging_loss = tr_loss

                    if tb_helper:
                        tb_helper(step=global_step, last_batch=batch, last_outputs=outputs)

                # Save model checkpoint
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                    if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(model, cfg, output_dir, tokenizer)

                # Evaluation
                if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    state_dict = get_state_dict(model, cfg)

                    if cfg.local_rank in [-1, 0]:
                        results = evaluate(cfg, model, tokenizer, prefix=str(global_step), _split="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar(f"eval/{key}", value, global_step)

                        sub_path = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                        flag = note_best_checkpoint(cfg, results, sub_path)
                        if cfg.save_best and flag:
                            save_model(model, cfg, cfg.output_dir, tokenizer, state_dict)
                            del state_dict

            if 0 < cfg.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < cfg.max_steps < global_step:
            train_iterator.close()
            break

    if cfg.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

```





