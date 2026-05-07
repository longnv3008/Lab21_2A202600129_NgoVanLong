# Lab 21 — Submission Links

**Họ và tên**: Ngô Văn Long  
**Mã học viên**: 2A202600129

---

## GitHub Repository

https://github.com/longnv3008/Lab21_2A202600129_NgoVanLong

## HuggingFace Hub — Fine-tuned Adapter (r=16)

https://huggingface.co/longnv3008/lab21-qwen2.5-3b-lora-r16

### Model Details

| Field | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen 2.5, 3B params, NF4 4-bit) |
| LoRA rank | r=16 (best ROI — sweet spot giữa chất lượng và VRAM) |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `v_proj` |
| Dataset | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples |
| Perplexity | 4.55 (↓ 30.8% vs base 6.58) |
| Peak VRAM | 6.62 GB |
| Training time | ~4 min on Tesla T4 |
| Training cost | ~$0.02 (@ $0.35/hr) |
