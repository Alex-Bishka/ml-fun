import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformer_lens import HookedTransformer
from datasets import load_dataset
import os


MODEL_NAME = "google/gemma-2-2b-it"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")
loaded_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, # <-- Note: we use the original name for the architecture
    torch_dtype=torch.bfloat16
)
state_dict = torch.load("./models/gemma-2-baseline.pth", weights_only=True)
loaded_model.load_state_dict(state_dict)

loaded_model.to(device)
loaded_model.eval()

print("Everything loaded - time to answer!\n")

text = "Hello, what is the capital of California?"
inputs = loaded_tokenizer(text, return_tensors="pt").to(device)
outputs = loaded_model.generate(**inputs, max_new_tokens=20)

print(loaded_tokenizer.decode(outputs[0], skip_special_tokens=True))

from torchinfo import summary
summary(loaded_model)

# ===========================================================================
# Layer (type:depth-idx)                             Param #
# ===========================================================================
# Gemma2ForCausalLM                                  --
# ├─Gemma2Model: 1-1                                 --
# │    └─Embedding: 2-1                              589,824,000
# │    └─ModuleList: 2-2                             --
# │    │    └─Gemma2DecoderLayer: 3-1                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-2                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-3                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-4                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-5                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-6                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-7                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-8                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-9                77,865,984
# │    │    └─Gemma2DecoderLayer: 3-10               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-11               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-12               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-13               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-14               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-15               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-16               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-17               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-18               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-19               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-20               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-21               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-22               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-23               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-24               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-25               77,865,984
# │    │    └─Gemma2DecoderLayer: 3-26               77,865,984
# │    └─Gemma2RMSNorm: 2-3                          2,304
# │    └─Gemma2RotaryEmbedding: 2-4                  --
# ├─Linear: 1-2                                      589,824,000
# ===========================================================================
# Total params: 3,204,165,888
# Trainable params: 3,204,165,888
# Non-trainable params: 0
# ===========================================================================