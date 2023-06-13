'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-06-13 20:59:16
'''
from huggingface_hub import snapshot_download
is_7b = False
is_13b = False


# | Panda-7B        | 7B         | https://huggingface.co/chitanda/llama-panda-zh-7b-delta  |
# | Panda-Instruct-7B | 7B       | https://huggingface.co/chitanda/llama-panda-zh-coig-7b-delta|
# | Panda-13B       | 13B        | https://huggingface.co/chitanda/llama-panda-zh-13b-delta |
# | Panda-Instruct-13B | 13B     | https://huggingface.co/chitanda/llama-panda-zh-13b-coig-delta|
# | Flan-LLaMA-7B   | 7B         | https://huggingface.co/NTU-NLP-sg/flan-llama-7b-10m-delta |

if is_7b:
    print("download origin Llama 7b model weight...")
    snapshot_download(repo_id="decapoda-research/llama-7b-hf")
    print("download deleta pandaLlama 7b model weight...")
    snapshot_download(repo_id="chitanda/llama-panda-zh-7b-delta")
if is_13b:
    print("download origin Llama 13b model weight...")
    snapshot_download(repo_id="decapoda-research/llama-13b-hf")
    print("download deleta pandaLlama 13b model weight...")
    snapshot_download(repo_id="chitanda/llama-panda-zh-13b-delta")

