import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 模型路径 - 根据实际情况修改
model_path = "/gemini/code/envs/Qwen"

# 检查CUDA可用性
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

try:
    # 加载分词器
    print(f"正在从 {model_path} 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 确定模型精度
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16
        print("使用bf16精度（GPU加速）")
    else:
        model_dtype = torch.float32
        print("使用float32精度（CPU模式）")

    # 加载模型
    print(f"正在从 {model_path} 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    print("模型加载成功！")

    # 创建文本生成流
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 测试提示语
    prompt = "请简要介绍一下人工智能的发展历程"
    print(f"\n提示: {prompt}")
    print("回答: ", end="")

    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成参数设置
    generate_config = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
        "do_sample": True
    }

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            streamer=streamer,
            **generate_config
        )

    # 解码并打印完整响应
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n\n完整响应:", response)

except Exception as e:
    print(f"发生错误: {e}")