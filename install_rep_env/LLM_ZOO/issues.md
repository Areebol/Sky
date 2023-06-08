<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-07 17:30:13
-->

# install 
**gpu017安装flash-attn会出现错误**
[解决方案](https://github.com/HazyResearch/flash-attention/issues/2466)
添加nvcc编译的环境变量
选择版本flash-attn==1.0.5

**transformers版本问题**
>ImportError: cannot import name 'LlamaTokenizer' from 'transformers'
[解决方案](https://github.com/huggingface/transformers/issues/22222)
选择特定的transformers分支
`pip install git+https://github.com/huggingface/transformers`

**auto-gptq安装问题**
[源码安装](https://github.com/PanQiWei/AutoGPTQ.git)
直接使用auto-gptq的源码安装
官方提供的方法
`BUILD_CUDA_EXT=0 pip install auto-gptq[triton]`

# running
**推理时无法精确设置cuda**
在实验室推理的时候，需要cuda0没有人占用，否则直接爆内存


# test
可以在cpu上进行int8的量化推理

