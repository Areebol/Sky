<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-08 21:28:14
-->

# Panda
基于Llama-7B -13B -33B -65B进行中文领域的持续预训练

在中文benchmark进行推理能力评测

**数据开源**
**模型开源**

暂时开放 Panda-7B Panda-13B
模型都可以在huggingface上下载
- 由于LLaMA权重License的限制，需要使用脚本来转化权重

## 数据
开源的中英文语料数据集
### 中文instruction-tuning

- Chinese Open Instruction Generalist (COIG)- [维基百科(wiki2019zh)，100万个结构良好的中文词条](https://github.com/brightmart/nlp_chinese_corpus)  
- [新闻语料(news2016zh)，250万篇新闻，含关键词、描述](https://github.com/brightmart/nlp_chinese_corpus)  
- [百科问答(baike2018qa)，150万个带问题类型的问答](https://github.com/brightmart/nlp_chinese_corpus)  
- [社区问答json版(webtext2019zh)，410万个高质量社区问答，适合训练超大模型](https://github.com/brightmart/nlp_chinese_corpus)  
- [翻译语料(translation2019zh)，520万个中英文句子对](https://github.com/brightmart/nlp_chinese_corpus)  
- [Chinese Open Instruction Generalist (COIG)](https://huggingface.co/datasets/BAAI/COIG) 

### 英文instruction-tuning
- FLAN Collection训练

### 训练框架
Deepspeed + gradient checkpointing

### 模型训练

对应模型的训练时超参数见：
```
# LLaMA-7b pretrain on general Chinese Corpus
conf/llama/zh/llama_7b_zh_instruct_v1_0_ds.yaml

# LLaMA-7b instruction tuning on COIG
conf/llama/zh/llama_7b_zh_instruct_coig_sft_v1_0_ds.yaml

# LLaMA-13b pretrain on general Chinese Corpus
conf/llama/zh/llama_13b_zh_instruct_v1_0_ds.yaml
```

Command:
```
HYDRA_FULL_ERROR=1 deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mul.py -cp conf/llama/zh -cn <file name of yaml config> 
```
我们的训练使用了 2 * 8 * A100 80G GPU。如使用更少的显卡，请相应的调整 `gradient_accumulation_steps` 和 `per_gpu_train_batch_size`.