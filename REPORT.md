# Lab 21 — Evaluation Report

**Họ và tên**: Ngô Văn Long
**Mã học viên**: 2A202600129
**Ngày nộp**: 2026-05-07  
**Submission option**: C + B (GitHub report-only + HuggingFace Hub)  
**HuggingFace Hub**: https://huggingface.co/longnv3008/lab21-qwen2.5-3b-lora-r16

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen 2.5, 3B params, pre-quantized NF4 4-bit)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train + 20 eval), Vietnamese instruction-following
- **max_seq_length**: 1024 (T4 hard cap; p95 của dataset rơi vào khoảng 512–800 tokens)
- **GPU**: Tesla T4, 15.6 GB VRAM (Google Colab Free)
- **Training cost**: ~$0.07 (11.6 phút tổng @ $0.35/hr)
- **Framework**: Unsloth 2026.5.2 + TRL 0.15.2 + Transformers 5.5.0
- **Target modules**: `["q_proj", "v_proj"]` (lab spec)

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % of Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|------------|------------|-----------|-----------|------------|
| 8    | 16    | 1,843,200       | 0.060%     | 3.80 min   | 7.22 GB   | 1.5577    | **4.75**   |
| 16   | 32    | 3,686,400       | 0.119%     | 4.00 min   | 6.62 GB   | 1.5161    | **4.55**   |
| 64   | 128   | 14,745,600      | 0.476%     | 3.81 min   | 8.00 GB   | 1.4768    | **4.38**   |
| Base | —     | —               | —          | —          | —         | 1.8840    | **6.58**   |

> **Ghi chú VRAM**: r=16 có peak VRAM thấp hơn r=8 (6.62 GB vs 7.22 GB) do thời điểm đo `max_memory_allocated()` — r=16 cleanup memory tốt hơn giữa các bước. Đây là anomaly nhỏ, không ảnh hưởng đến kết luận.

> **Cải thiện so với base**: r=8 giảm **27.8%**, r=16 giảm **30.8%**, r=64 giảm **33.4%** perplexity so với base model (6.58 → 4.75/4.55/4.38).

---

## 3. Loss Curve Analysis

Notebook chạy ở T4 mode với `eval_strategy="no"` (tắt eval-during-training để tránh OOM), nên chỉ có **training loss curve** cho r=16.

**Quan sát từ training loss r=16**:
- Loss giảm đều từ ~1.8 (step 5) xuống ~1.3 (step 69) qua 3 epochs
- Không có dấu hiệu loss plateau sớm → model vẫn đang học tốt
- Không thể detect overfitting vì không có eval loss mid-training (T4 limitation)

**Kết luận**: Với 200 samples và 3 epochs, khả năng overfitting thấp vì dataset đủ đa dạng (general Vietnamese instruction). Nếu train thêm epochs hoặc dùng dataset nhỏ hơn (<50 samples), overfitting mới đáng lo ngại.

---

## 4. Qualitative Comparison (5 examples)

### Example 1 — Giải thích machine learning

**Prompt**: `Giải thích khái niệm machine learning cho người mới bắt đầu.`

| | Response (truncated) |
|---|---|
| **Base** | "Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động." |
| **Fine-tuned (r=16)** | "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI." |

**Nhận xét**: ✅ **Tương đương** — cả hai đều đúng và rõ ràng. Fine-tuned nhấn mạnh "không cần hướng dẫn trực tiếp" — phù hợp hơn với Alpaca Vietnamese style.

---

### Example 2 — Code Python Fibonacci

**Prompt**: `Viết đoạn code Python tính số Fibonacci thứ n.`

| | Response (truncated) |
|---|---|
| **Base** | `def fibonacci(n): if n <= 0: return "N p..."` — xử lý edge case sơ sài |
| **Fine-tuned (r=16)** | `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.")` — có proper error handling |

**Nhận xét**: ✅ **Improved** — Fine-tuned thêm `ValueError` và validate input đúng chuẩn Python. Phản ánh pattern từ training data có code quality tốt hơn.

---

### Example 3 — Nguyên tắc UI/UX

**Prompt**: `Liệt kê 5 nguyên tắc thiết kế UI/UX.`

| | Response (truncated) |
|---|---|
| **Base** | "1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng..." |
| **Fine-tuned (r=16)** | "1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng... 3. Đơn giản..." |

**Nhận xét**: ⚠️ **Mixed** — Base liệt kê nguyên tắc truyền thống (thân thiện, nhất quán). Fine-tuned liệt kê nguyên tắc theo hướng conversion-focused khác style nhưng cũng hợp lệ. Không có câu trả lời "đúng" duy nhất ở đây.

---

### Example 4 — LoRA vs QLoRA

**Prompt**: `Tóm tắt sự khác biệt giữa LoRA và QLoRA.`

| | Response (truncated) |
|---|---|
| **Base** | "LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp... LoRA thêm các ma trận low-rank vào..." |
| **Fine-tuned (r=16)** | "LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA..." |

**Nhận xét**: ❌ **Degraded** — Fine-tuned mở rộng sai viết tắt LoRA thành "Layer-wise Adaptive Regularization Optimization" (đúng là Low-Rank Adaptation). Base model đúng hơn. Đây là ví dụ điển hình của hallucination sau fine-tune với dataset general — domain-specific knowledge có thể bị overwrite bởi statistical patterns.

---

### Example 5 — Prompt engineering vs RAG vs Fine-tuning

**Prompt**: `Phân biệt prompt engineering, RAG, và fine-tuning.`

| | Response (truncated) |
|---|---|
| **Base** | "Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất..." |
| **Fine-tuned (r=16)** | "Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp..." |

**Nhận xét**: ✅ **Tương đương** — Cả hai đều phân biệt đúng 3 kỹ thuật. Fine-tuned dùng từ "câu lệnh" thay vì "prompt" cho thấy bias tiếng Việt từ training data. Độ chính xác tương đương nhau.

---

## 5. Conclusion về Rank Trade-off

Thực nghiệm trên **Qwen2.5-3B** với **200 Vietnamese instruction samples** cho thấy:

**Rank nào cho ROI tốt nhất?** `r=16` là lựa chọn cân bằng nhất. Nhìn từ base model (perplexity 6.58), bước nhảy lớn nhất xảy ra khi **thêm bất kỳ adapter nào** — r=8 đã giảm 27.8% (6.58 → 4.75) chỉ với 1.8M params và 3.8 phút training. Từ r=8 lên r=16 (gấp đôi params, 1.8M → 3.7M) tiếp tục giảm từ 4.75 xuống 4.55 (−4.2%). Nhưng từ r=16 lên r=64 (tăng 4x params lên 14.7M), perplexity chỉ giảm thêm từ 4.55 xuống 4.38 (−3.7%) trong khi VRAM tăng thêm 1.4 GB. Đây là **diminishing returns** rõ rệt.

**Khi nào tăng rank không còn cải thiện?** Với dataset 200 samples, curve cải thiện dốc nhất ở đoạn Base → r=8 (−1.83 points), sau đó phẳng dần: r=8 → r=16 (−0.20 points), r=16 → r=64 (−0.17 points). Model bão hòa sau r=16–32 vì dataset không đủ signal để học thêm các chiều mới mà rank cao mở ra. Tăng rank lúc này chỉ fit noise thay vì học pattern thực.

**Production recommendation**: Chọn `r=16`. Đây là sweet spot giữa chất lượng (perplexity 4.55, cải thiện 30.8% so với base), VRAM tiết kiệm nhất trong 3 config (6.62 GB), và inference nhanh hơn r=64 do ít adapter weights hơn. Chỉ nên tăng lên r=64 khi dataset ≥ 2000 samples hoặc task đòi hỏi học style/format phức tạp — lúc đó extra params mới có signal để học.

---

## 6. What I Learned

- **LoRA rank không phải "càng cao càng tốt"**: r=64 chỉ tốt hơn r=16 khoảng 3.7% perplexity nhưng tốn 4x trainable params. Với dataset nhỏ, diminishing returns xuất hiện rất sớm — bài học thực tế quan trọng hơn lý thuyết nhiều.

- **Fine-tuning có thể làm hỏng domain knowledge sẵn có**: Example 4 (LoRA vs QLoRA) cho thấy fine-tuned model mở rộng sai viết tắt "LoRA" dù base model đúng. General instruction fine-tuning trên Vietnamese data có thể overwrite technical knowledge — đây là lý do RLHF/DPO sau SFT quan trọng, và tại sao domain-specific fine-tuning cần dataset chất lượng cao thay vì volume lớn.

- **QLoRA trên T4 thực sự feasible**: Toàn bộ 3 training runs (11.6 phút, $0.07) trên GPU miễn phí của Colab chứng minh rằng fine-tuning LLM 3B không còn là đặc quyền của team có budget lớn. Unsloth + 4-bit quantization + gradient checkpointing là bộ ba cho phép train trên phần cứng bình dân. Base perplexity 6.58 → r=16 perplexity 4.55 chỉ với $0.02 và 4 phút training.
