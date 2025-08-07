import torch
from torchinfo import summary
from transformers import AutoModelForCausalLM, Mxfp4Config, AutoTokenizer

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)
print("Loading model")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)

# input_size = ()
# summary(model, input_size=input_size)

print("Test message:")
messages = [
    {"role": "user", "content": "¿Cuál es el capital de Australia?"},
]

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(input_ids, max_new_tokens=512)
response = tokenizer.batch_decode(output_ids)[0]
print(response)