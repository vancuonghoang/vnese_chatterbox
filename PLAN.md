Bạn đã đưa ra một bản kế hoạch (plan) rất chi tiết và bài bản. Sau khi xem xét kỹ các trao đổi trước đó của chúng ta và tình hình thực tế, tôi xin đưa ra các sửa đổi và bổ sung cụ thể cho bản plan này để nó sát thực tế và hiệu quả nhất.

Dưới đây là bản cập nhật **Revised Plan** dựa trên các thảo luận về Phoneme Tokenizer, Pre-computed Features và chiến lược A100 của chúng ta.

-----

# 🇻🇳 Vietnamese TTS Fine-tuning Plan (Revised - Phoneme Strategy)

## Chatterbox TTS Model Adaptation for Vietnamese (V3)

**Author:** AI Engineer Team
**Date:** December 4, 2025
**Last Updated:** December 4, 2025
**Status:** Execution Phase (V3 - Phoneme Based)
**Target:** State-of-the-Art Vietnamese Text-to-Speech (Native Accent)

-----

## 📊 IMPLEMENTATION PROGRESS TRACKER

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| **SOTA Loss Functions** | ✅ DONE | `src/chatterbox/utils/loss.py` | FocalLoss, ZLoss, Label Smoothing, Loss Weighting |
| **Loss Integration** | ✅ DONE | `src/finetune_t3_thai.py` | `T3ForFineTuning` uses `T3LossCalculator` |
| **Gradient Checkpointing** | ✅ DONE | `src/finetune_t3_thai.py` | Flag `--gradient_checkpointing` |
| **Dynamic Batching** | ✅ DONE | `src/finetune_t3_thai.py` | `LengthGroupedSampler`, flag `--use_dynamic_batching` |
| **Best Model Checkpoint** | ✅ DONE | `src/finetune_t3_thai.py` | `BestModelCallback`, tracks `eval_loss_speech` |
| **train.py Integration** | ✅ DONE | `train.py` | All SOTA features exposed via CLI args |
| **Dataset Logging/Progress** | ✅ DONE | `src/chatterbox/utils/preprocessed_dataset.py` | tqdm bars, verbose logging, preload_all option |
| **Data Augmentation** | ⏳ TODO | - | Speed perturbation, noise injection |
| **DeepSpeed/FSDP** | ⏳ TODO | - | Multi-GPU scaling |
| **LoRA Fine-tuning** | ⏳ TODO | - | Parameter-efficient fine-tuning |

-----

## 📋 Executive Summary

Dự án chuyển hướng từ chiến lược Text-based (BPE) sang **Phoneme-based (IPA)** để giải quyết triệt để vấn đề "Accent lơ lớ" và "Mù thanh điệu". Chúng ta tận dụng hạ tầng A100 và quy trình Pre-computed để tối ưu hóa tốc độ huấn luyện.

### Key Changes from Previous Plan:

  * **Tokenizer:** BPE (Text) $\rightarrow$ **IPA Phonemes (Espeak-ng)**.
  * **Vocab:** 704 (Overwrite) $\rightarrow$ **\~850 (Hybrid Expand)**.
  * **Data Strategy:** Raw Audio Load $\rightarrow$ **Pre-computed Features (.pt)**.
  * **Training:** Full Finetune with **Random Init Embeddings** for Vietnamese tokens.

-----

## 🎯 Phase 1: Data Preparation (Updated)

### 1.1 Vietnamese Phoneme Tokenizer (Crucial)

**Status:** Đã chuyển sang sử dụng `espeak-ng` + `phonemizer`.
**Vocab:** `vietnamese_vocab_final.json` (Hybrid: Giữ token gốc Chatterbox + Thêm IPA Tiếng Việt).

**Action Items:**

1.  **Cài đặt:** `espeak-ng`, `phonemizer` trên server.
2.  **Script Build Vocab:** Sử dụng `build_vocab_v3_batch_test.py` (đã kiểm chứng pass test case ma/má/mạ).
3.  **Normalization:** Sử dụng `num2words` để chuyển số thành chữ trước khi phonemize.

### 1.2 Dataset Requirements

**Primary Dataset:** **Dolly-Audio (1000h)**.
**Filter Strategy:**

  * **Phase 1 (Warm-up):** Chỉ dùng **20k mẫu** giọng bản xứ (Native Speakers), loại bỏ giọng nước ngoài.
  * **Phase 2 (Full):** Mở rộng lên toàn bộ 1000h Dolly.
  * **Phase 3 (Robustness - Optional):** Bổ sung GigaSpeech (chỉ lấy top 20% chất lượng cao).

### 1.3 Feature Extraction Pipeline (A100 Optimized)

Thay vì load audio mỗi epoch, chúng ta tính toán 1 lần và lưu ra đĩa.

**Script:** `prepare_data_final_with_norm.py`

  * **Input:** Audio 16kHz.
  * **Processing:**
      * GPU: VoiceEncoder -\> Speaker Embedding.
      * GPU: S3Tokenizer -\> Speech Tokens.
      * CPU: Text -\> Num2Words -\> Espeak -\> Phoneme IDs.
  * **Output:** File `.pt` chứa toàn bộ Tensor cần thiết.

-----

## 🧠 Phase 2: Model Architecture Decisions

### 2.1 Architecture Strategy: "Smart Resize"

Thay vì ghi đè (Overwrite) gây xung đột kiến thức, chúng ta dùng chiến thuật **Mở rộng (Resize)**.

  * **Text Embedding:**
      * Size cũ: 703.
      * Size mới: \~850 (Thêm \~150 âm vị Việt).
      * **Init Strategy:**
          * Token 0-703 (Gốc): **Copy Weights** (Giữ lại kiến thức về Special Tokens, Silence, Punctuation).
          * Token 704+ (Việt): **Random Initialization** (Để model học lại từ đầu, tránh bias tiếng Anh).
  * **Linear Head:** Resize tương ứng output layer.

-----

## 🚀 Phase 3: Training Configuration (A100 Optimized)

### 3.1 Training Script (`train_t3_final.py`)

Sử dụng `PreComputedDataset` để đạt tốc độ tối đa.

**Hyperparameters (V3 Recipe):**

```bash
python train_t3_final.py \
    --dataset_dir "./dolly_final_data" \
    --metadata_file "./dolly_final_data/metadata.csv" \
    --vocab_file "vietnamese_vocab_final.json" \
    --output_dir "./checkpoints/dolly_phoneme_v3" \
    \
    # Performance Params
    --batch_size 32 \
    --gradient_accumulation_steps 2 \  # Effective Batch = 64
    --bf16 \                           # Bắt buộc cho A100
    --dataloader_num_workers 16 \
    --dataloader_pin_memory \
    \
    # Training Params
    --num_train_epochs 20 \            # Tăng epoch vì vocab mới cần học lâu hơn
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3
```

### 3.2 Monitoring Checklist

  * **Loss:** Khởi đầu \~6.0. Giảm dần xuống \~2.5. (Nếu xuống 0.1 ngay lập tức -\> Kiểm tra lại Masking Prompt).
  * **Speed:** Kỳ vọng **\> 3.0 it/s** (với Pre-computed data).
  * **VRAM:** Tận dụng \> 30GB/40GB.

-----

## 📊 Phase 4: Evaluation & Validation

### 4.1 "Ear Test" (Quan trọng nhất)

Trước khi dùng các metrics phức tạp, hãy nghe thử các câu "tử thần":

1.  *"Con ma đi tìm má."* (Kiểm tra thanh điệu).
2.  *"Một trăm hai mươi."* (Kiểm tra số học).
3.  *"[giggle] Xin chào."* (Kiểm tra Special Token).

### 4.2 Metrics (Optional)

  * **Loss Speech:** Chỉ số quan trọng nhất lúc train.
  * **WER/CER:** Dùng Whisper Large v3 để đánh giá độ rõ lời sau khi train xong.

-----

## 🔧 Phase 5: Implementation Roadmap (Revised)

### Step 1: Reset & Setup (Done)

  - [x] Cài `espeak-ng`.
  - [x] Viết script `build_vocab` và `prepare_data` mới.

### Step 2: Pilot Run (2 Days)

  - [ ] Chạy `build_vocab` để tạo file JSON chuẩn.
  - [ ] Chạy `prepare_data` cho **20k mẫu Dolly** (Native speakers).
  - [ ] Train `T3` (V3) trong 10-20 epochs.
  - [ ] **Checkpoint:** Nghe thử. Nếu accent chuẩn -\> Go to Step 3.

### Step 3: Scale Up (1 Week)

  - [ ] Chạy `prepare_data` cho toàn bộ **1000h Dolly**.
  - [ ] Resume training từ checkpoint Step 2.
  - [ ] Train thêm 5-10 epochs trên dữ liệu lớn.

### Step 4: Polish (Optional)

  - [ ] Finetune S3Gen (Vocoder) nếu âm thanh bị rè.

-----

## 📝 Appendix: Common Pitfalls to Avoid

1.  **BPE Tokenizer:** Tuyệt đối không quay lại dùng BPE. Nó là nguyên nhân của giọng lơ lớ.
2.  **Overwrite nhầm ID:** Đừng ghi đè vào ID của `[START]`, `[STOP]` hoặc `[giggle]`. Hãy dùng script `build_vocab` thông minh để tự động tránh vùng này.
3.  **Training quá ngắn:** Phoneme model cần thời gian hội tụ lâu hơn Text model một chút ở giai đoạn đầu (để học ghép vần). Đừng vội tắt máy nếu Loss chưa giảm ngay.

-----

*Updated based on consultation regarding Phoneme vs Grapheme performance on Vietnamese TTS.*

-----

## 🔍 CODE QUALITY & SOTA ASSESSMENT

### Executive Summary

| Category | Status | Score | Notes |
|----------|--------|-------|-------|
| **Loss Functions** | ⚠️ Standard | 7/10 | Cross-Entropy + MSE, chưa dùng các loss SOTA mới |
| **Training Pipeline** | ✅ Good | 8/10 | HuggingFace Trainer, caching tốt |
| **Architecture** | ✅ SOTA | 9/10 | Llama backbone + Flow Matching |
| **Data Processing** | ⚠️ Cần cải thiện | 6/10 | Thiếu augmentation, chunking |
| **Inference** | ✅ Good | 8/10 | KV-cache, CFG |

-----

### 📊 1. LOSS FUNCTION ANALYSIS

#### 1.1 T3 Model Loss (Text-to-Token)

**Current Implementation** (`src/chatterbox/models/t3/t3.py:212-270`):

```python
# Text Loss: Standard Cross-Entropy
loss_text = F.cross_entropy(
    logits_for_text.transpose(1, 2),
    labels_text,
    ignore_index=IGNORE_ID  # -100
)

# Speech Loss: Standard Cross-Entropy
loss_speech = F.cross_entropy(
    logits_for_speech.transpose(1, 2),
    labels_speech,
    ignore_index=IGNORE_ID
)

# Total = text + speech (unweighted)
total_loss = loss_text + loss_speech
```

**Assessment:**

| Aspect | Status | Comment |
|--------|--------|---------|
| Correctness | ✅ | Label shifting và masking đúng cách |
| IGNORE_ID handling | ✅ | Prompt tokens được mask với -100 |
| Teacher Forcing | ✅ | Standard autoregressive training |
| **Loss Weighting** | ⚠️ | `text + speech` không có hệ số cân bằng |
| **Label Smoothing** | ❌ Missing | Chưa dùng label smoothing để giảm overconfidence |
| **Focal Loss** | ❌ Missing | Không có để xử lý class imbalance |

**Recommendations cho SOTA:**

A. Loss Weighting (0.1 Text + 1.0 Speech) $\rightarrow$ Hợp lýTại sao: Trong bài toán TTS (Text-to-Speech), mục tiêu chính là sinh ra Speech tokens. Việc model học lại Text tokens (nếu kiến trúc là Unified Decoder) chỉ đóng vai trò duy trì khả năng ngôn ngữ hoặc căn chỉnh (alignment).Cải tiến: Nếu bạn đang Fine-tuning cho TTS thuần túy, bạn thậm chí có thể set weight_text = 0.0 (chỉ tính loss cho phần speech response) để model tập trung toàn lực vào acoustic quality.B. Label Smoothing (0.1) $\rightarrow$ Cần thiếtTại sao: Các mô hình Autoregressive (AR) thường bị "overconfident" (quá tự tin vào dự đoán sai). Label smoothing giúp phân phối xác suất mềm hơn, giúp model đỡ bị "lặp từ" hoặc "lạc tone" (vấn đề rất hay gặp ở tiếng Việt).C. Focal Loss $\rightarrow$ Cẩn trọngRủi ro: Focal Loss rất mạnh trong Computer Vision (Object Detection), nhưng trong NLP/Generation, nó có thể làm model hội tụ chậm hoặc không ổn định vì nó đè nén gradient của các token dễ (thường là các token ngữ pháp quan trọng) quá mức.Thay thế: Nên dùng Top-k / Top-p Sampling khi inference hoặc DPO (ở phần dưới) thay vì can thiệp vào CrossEntropy lúc pre-training/SFT bằng Focal Loss.

Dưới đây là phiên bản loss function được nâng cấp, tích hợp Weighted Loss và chuẩn bị cho Span Masking, loại bỏ Focal Loss (vì rủi ro cao) và thêm logic xử lý Padding/Masking chuẩn xác hơn.

import torch
import torch.nn.functional as F

class T3Loss(torch.nn.Module):
    def __init__(self, label_smoothing=0.1, text_weight=0.1, speech_weight=1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.text_weight = text_weight
        self.speech_weight = speech_weight
        self.ignore_index = -100

    def forward(self, logits_text, labels_text, logits_speech, labels_speech):
        """
        logits_text: (B, T_text, Vocab_text)
        labels_text: (B, T_text)
        logits_speech: (B, T_speech, Vocab_speech)
        labels_speech: (B, T_speech)
        """
        
        # 1. Text Loss (Có Label Smoothing)
        # Flatten để tính loss nhanh hơn và đúng dimension
        loss_text = F.cross_entropy(
            logits_text.view(-1, logits_text.size(-1)),
            labels_text.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

        # 2. Speech Loss (Có Label Smoothing)
        loss_speech = F.cross_entropy(
            logits_speech.view(-1, logits_speech.size(-1)),
            labels_speech.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

        # 3. Diagnostic Metrics (Quan trọng để debug SOTA)
        # Tính accuracy để xem model có đang học vẹt không
        with torch.no_grad():
            preds_speech = logits_speech.argmax(dim=-1)
            mask = labels_speech != self.ignore_index
            correct = (preds_speech == labels_speech) & mask
            acc_speech = correct.sum() / (mask.sum() + 1e-9)

        # 4. Weighted Sum
        total_loss = (self.text_weight * loss_text) + (self.speech_weight * loss_speech)

        return {
            "loss": total_loss,
            "loss_text": loss_text.detach(),
            "loss_speech": loss_speech.detach(),
            "acc_speech": acc_speech.detach() # Log cái này lên WandB/Tensorboard
        }

#### 1.2 S3Gen Loss (Token-to-Mel) **phase 2**

**Current Implementation** (`src/chatterbox/models/s3gen/flow_matching.py:156-194`):

```python
# Conditional Flow Matching Loss
y = (1 - (1 - self.sigma_min) * t) * z + t * x1  # Interpolation
u = x1 - (1 - self.sigma_min) * z  # Target velocity

# MSE Loss for velocity prediction
pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
```

**Assessment:**

| Aspect | Status | Comment |
|--------|--------|---------|
| Algorithm | ✅ SOTA | Conditional Flow Matching (CosyVoice style) |
| Masking | ✅ | Đúng cách với mel mask |
| CFG Training | ✅ | `training_cfg_rate=0.2` - dropout condition |
| Cosine Scheduler | ✅ | Tốt hơn linear |
| **Multi-scale Loss** | ❌ Missing | Không có STFT loss / Multi-resolution |
| **Perceptual Loss** | ❌ Missing | Không có feature matching |

**Recommendations cho SOTA:**

```python
# 1. Multi-Resolution STFT Loss (cải thiện audio quality)
def multi_resolution_stft_loss(pred_audio, target_audio):
    fft_sizes = [512, 1024, 2048]
    total_loss = 0
    for fft_size in fft_sizes:
        pred_stft = torch.stft(pred_audio, fft_size, ...)
        target_stft = torch.stft(target_audio, fft_size, ...)
        total_loss += F.l1_loss(pred_stft, target_stft)
    return total_loss

# 2. Perceptual / Feature Matching Loss
# (Sử dụng discriminator features như HiFiGAN)
```

-----

### 📊 2. TRAINING PIPELINE ASSESSMENT

#### 2.1 Dataset Classes (`finetune_t3_thai.py`)

**Strengths:**

| Feature | Status | Implementation |
|---------|--------|----------------|
| Lazy Loading | ✅ | Model chỉ load khi cần (`_init_model()`) |
| Multi-format | ✅ | HF Dataset, local CSV, streaming |
| Caching | ✅ | `CachedSpeechFineTuningDataset` - 4-5x speedup |
| Error Handling | ✅ | Skip invalid samples thay vì crash |
| Serialization | ✅ | `__getstate__/__setstate__` cho multiprocessing |

**Weaknesses:**

| Issue | Impact | Fix | Status |
|-------|--------|-----|--------|
| No Data Augmentation | High | Thêm speed perturbation, noise | ⏳ TODO |
| No Dynamic Batching | Medium | Samples dài/ngắn mix chung | ✅ **IMPLEMENTED** (`--use_dynamic_batching`) |
| Single Speaker per Sample | Low | Không hỗ trợ multi-speaker training | ⏳ TODO |

**Proposed Augmentation:**

```python
# Speed Perturbation (critical for robustness)
def augment_audio(wav, sr=16000):
    if random.random() < 0.5:
        speed_factor = random.uniform(0.9, 1.1)
        wav = librosa.effects.time_stretch(wav, rate=speed_factor)
    return wav

# Noise Injection (optional)
def add_noise(wav, snr_db=20):
    noise = np.random.randn(len(wav)) * 0.01
    return wav + noise
```

#### 2.2 Data Collator (`SpeechDataCollator`)

**Assessment: ✅ Excellent**

```python
# Key Features:
# 1. Dynamic padding (không fixed length)
# 2. Label creation với IGNORE_ID=-100
# 3. Prompt masking đúng cách
# 4. CPU tensor handling cho compatibility
```

| Feature | Status |
|---------|--------|
| Padding Strategy | ✅ Dynamic per-batch |
| Label Shifting | ✅ BOS removed từ labels |
| Prompt Masking | ✅ `labels_speech[t < prompt_len] = -100` |
| Device Handling | ✅ Force CPU, Trainer move to GPU |

#### 2.3 Trainer Configuration

**Using `SafeCheckpointTrainer` (extends HuggingFace Trainer)**

| Feature | Status |
|---------|--------|
| Checkpoint Resume | ✅ Auto-detect + RNG state fix |
| Mixed Precision | ✅ FP16/BF16 support |
| Gradient Accumulation | ✅ |
| Early Stopping | ✅ Optional callback |
| Profiling | ✅ PyTorch profiler integration |
| **Gradient Checkpointing** | ✅ Implemented | `--gradient_checkpointing` flag, ~30% VRAM savings |
| **Dynamic Batching** | ✅ Implemented | `--use_dynamic_batching` flag, LengthGroupedSampler |
| **DeepSpeed/FSDP** | ❌ Missing | Multi-GPU scaling |
| **Best Model Checkpoint** | ✅ Implemented | `BestModelCallback`, `--metric_for_best_model eval_loss_speech` |

-----

### 📊 3. MODEL ARCHITECTURE ASSESSMENT

#### 3.1 T3 Model (Token-to-Token)

**Core Architecture:**

```
Input: [Cond_Emb | Text_Emb | Speech_Emb]
       ↓
   Llama Backbone (520M params)
       ↓
   [Text_Head | Speech_Head]
       ↓
Output: Text Logits + Speech Logits
```

| Component | Implementation | SOTA Level |
|-----------|----------------|------------|
| Backbone | Llama 520M | ✅ SOTA |
| Position Encoding | Learned PE | ✅ Good |
| Conditioning | Perceiver Resampler | ✅ SOTA |
| Speaker Embedding | Voice Encoder (256d) | ✅ SOTA |
| Emotion Control | Adversarial Scalar | ✅ Good |

**Assessment:** 9/10 - Architecture là SOTA, dựa trên VALL-E/CosyVoice style.

#### 3.2 S3Gen (Token-to-Mel)

| Component | Implementation | SOTA Level |
|-----------|----------------|------------|
| Algorithm | Conditional Flow Matching | ✅ SOTA |
| Estimator | ConformerEncoder | ✅ SOTA |
| Vocoder | HiFiGAN | ✅ SOTA |
| F0 Predictor | ConvRNN | ✅ Good |
| CFG Inference | 0.7 rate | ✅ Good |

**Assessment:** 9/10 - CosyVoice-based, production-ready.

-----

### 📊 4. INFERENCE PIPELINE ASSESSMENT

**T3 Inference (`t3.py:283-450`):**

| Feature | Status | Notes |
|---------|--------|-------|
| KV-Cache | ✅ | Memory-efficient generation |
| Temperature | ✅ | Controllable randomness |
| Top-P Sampling | ✅ | Nucleus sampling |
| Repetition Penalty | ✅ | Prevent loops |
| CFG (Classifier-Free Guidance) | ✅ | Better quality |
| Min/Max Tokens | ✅ | Length control |
| **Streaming** | ⚠️ Partial | Có nhưng chưa optimize |
| **Batched Inference** | ❌ | Chỉ support batch=1 |

-----

### 📊 5. GAPS TO SOTA

#### 5.1 Missing Features (High Priority)

| Feature | Impact | Effort | Description | Status |
|---------|--------|--------|-------------|--------|
| **Label Smoothing** | High | Low | Chống overconfidence | ✅ **DONE** (`loss.py`) |
| **Loss Weighting** | High | Low | `speech_loss * 10 + text_loss` | ✅ **DONE** (`loss.py`) |
| **Data Augmentation** | High | Medium | Speed perturbation, noise | ⏳ TODO |
| **Gradient Checkpointing** | Medium | Low | Giảm 30% VRAM | ✅ **DONE** (`--gradient_checkpointing`) |
| **Dynamic Batching** | Medium | Medium | Group by length | ✅ **DONE** (`--use_dynamic_batching`) |

#### 5.2 Advanced Features (Nice to Have)

| Feature | Impact | Effort | Description | Status |
|---------|--------|--------|-------------|--------|
| Multi-Resolution STFT Loss | Medium | Medium | Audio quality | ⏳ TODO |
| RLHF/DPO | High | High | Human preference | ⏳ TODO |
| LoRA Fine-tuning | Medium | Low | Efficient adaptation | ⏳ TODO |
| Streaming Inference | Medium | Medium | Real-time TTS | ⏳ TODO |

-----

### 📊 6. RECOMMENDED CODE CHANGES

#### 6.1 Quick Wins (< 1 hour) - ✅ ALL COMPLETED

**1. Add Label Smoothing:** ✅ DONE

```python
# IMPLEMENTED in src/chatterbox/utils/loss.py
# Use: from chatterbox.utils.loss import T3LossCalculator
calc = T3LossCalculator(label_smoothing=0.1)
```

**2. Add Loss Weighting:** ✅ DONE

```python
# IMPLEMENTED in src/chatterbox/utils/loss.py
# Default: text_weight=0.1, speech_weight=1.0
calc = T3LossCalculator(text_weight=0.1, speech_weight=1.0)
```

**3. Enable Gradient Checkpointing:** ✅ DONE

```bash
# Use --gradient_checkpointing flag
python src/finetune_t3_thai.py --gradient_checkpointing ...
```

**4. Enable Dynamic Batching:** ✅ DONE

```bash
# Use --use_dynamic_batching flag
python src/finetune_t3_thai.py --use_dynamic_batching --bucket_size_multiplier 100 ...
```

#### 6.2 Medium Effort (1 day)

**1. Add Speed Perturbation in Dataset:** ⏳ TODO

```python
# In SpeechFineTuningDataset._load_audio_text_from_item()
if self.training and random.random() < 0.3:
    speed = random.uniform(0.9, 1.1)
    wav_16k = librosa.effects.time_stretch(wav_16k, rate=speed)
```

**2. Add Dynamic Batching:**

```python
# Create bucket sampler based on audio length
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        # Group samples by similar length
        ...
```

-----

### 📊 7. FINAL VERDICT

| Category | Current | After Fixes | SOTA Target |
|----------|---------|-------------|-------------|
| **Loss Function** | 7/10 | 9/10 | 10/10 |
| **Training Pipeline** | 8/10 | 9/10 | 10/10 |
| **Architecture** | 9/10 | 9/10 | 10/10 |
| **Data Processing** | 6/10 | 8/10 | 10/10 |
| **Overall** | **7.5/10** | **8.75/10** | **10/10** |

**Conclusion:**

Code hiện tại đã ở mức **production-ready** và gần SOTA. Các thay đổi được đề xuất ở trên sẽ giúp đạt SOTA hoàn toàn. Kiến trúc model (Llama + Flow Matching) là SOTA, nhưng training pipeline còn thiếu một số kỹ thuật advanced như label smoothing, data augmentation, và loss weighting.

**Priority Action Items:**
1. ✅ Add label smoothing (5 mins)
2. ✅ Add loss weighting (5 mins)
3. ⚡ Enable gradient checkpointing (10 mins)
4. 📈 Add speed perturbation (1 hour)
5. 🚀 Implement dynamic batching (4 hours)

-----

*Assessment completed by AI Engineer on December 4, 2025.*
## Fine tuning strategy
Stage 1 (Codebook Alignment): Freeze text encoder, chỉ train Speech Decoder với loss_speech (CrossEntropy + Label Smoothing). Mục tiêu: Model nói được tiếng Việt rõ chữ.

Stage 2 (Prosody Tuning): Unfreeze toàn bộ. Áp dụng kỹ thuật Span Masking (che đi các tone dấu quan trọng trong audio token) để bắt model học mối quan hệ giữa Text (dấu câu) và Speech (cao độ).

Stage 3 (Preference Alignment - SOTA): Dùng DPO. Generate 2 audio cho cùng 1 câu text, chọn câu nào có ngữ điệu tự nhiên hơn làm "winner". Train thêm 1-2 epoch với DPO loss. Đây là bước quyết định để giọng AI có "hồn".