# 📘 Vietnamese TTS Training Guide (Pre-computed Flow)

Hướng dẫn training model Chatterbox TTS cho tiếng Việt sử dụng luồng **Pre-computed** (Tối ưu cho Production).

## 🚀 Workflow

```mermaid
graph LR
    A[Metadata CSV] --> B[preprocess_dataset.py]
    B --> C[Dataset đã xử lý (.pt files)]
    C --> D[train/run.py]
    D --> E[Checkpoints]
```

---

## 1. Cài đặt

Cài đặt các thư viện cần thiết cho training:

```bash
pip install -r requirements.txt
pip install -r viterbox-tts/requirements-train.txt
```

---

## 2. Chuẩn bị Dữ liệu

Tạo file `metadata.csv` với định dạng: `audio_path|transcript`.

**Ví dụ `metadata.csv`:**
```csv
audio|transcript
wavs/audio_001.wav|Xin chào các bạn.
wavs/audio_002.wav|Hôm nay trời đẹp quá.
/absolute/path/valid.wav|Đường dẫn tuyệt đối cũng được hỗ trợ.
```

> **Lưu ý:**
> - Audio nên là file WAV (mono). Sample rate sẽ tự động được convert về 16kHz.
> - Transcript nên là tiếng Việt có dấu.

---

## 3. Bước 1: Pre-processing (Xử lý dữ liệu)

Chạy script này để tính toán trước embeddings và tokens. Bước này giúp training nhanh hơn 5-10 lần.

**Chạy với GPU (Khuyên dùng - Nhanh nhất với batching):**
```bash
python preprocess_dataset.py \
    --metadata_csv metadata.csv \
    --audio_dir wavs \
    --output_dir ./preprocessed \
    --checkpoint ./vietnamese/pretrained_model_download \
    --device cuda \
    --batch_size 16 \
    --num_workers 1
```

**Tham số quan trọng:**
- `--metadata_csv`: Đường dẫn file metadata.
- `--audio_dir`: Thư mục chứa file audio.
- `--output_dir`: Thư mục lưu file `.pt` đã xử lý.
- `--batch_size`: Batch size cho GPU (default: 1, recommended: 8-16 cho GPU).
- `--num_workers`: Số luồng xử lý (Dùng 1 cho GPU để tránh OOM, dùng 4-8 cho CPU).

---

## 4. Bước 2: Training

Sử dụng `train/run.py` để training từ dữ liệu đã xử lý.

> **Lưu ý**: Script training nằm ở `train/run.py` (không phải `train_precomputed.py`)

### Cách 1: Fine-tuning tiêu chuẩn (Full Model)

Dành cho dataset lớn (>10h) hoặc khi cần chất lượng cao nhất.

```bash
python train/run.py \
    --preprocessed_dir ./preprocessed \
    --output_dir ./checkpoints/vietnamese_full \
    --epochs 20 \
    --batch_size 8 \
    --lr 5e-5 \
    --use_wandb
```

### Cách 2: LoRA Fine-tuning (Khuyên dùng cho 2-5h audio) ⭐ UPDATED

**Tối ưu mới (Tier 1)**: Tăng capacity và focus vào voice quality!

```bash
python train/run.py \
    --preprocessed_dir ./preprocessed \
    --output_dir ./checkpoints/vietnamese_lora \
    --epochs 20 \
    --batch_size 8 \
    --lr 5e-4 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --text_weight 0.05 \
    --speech_weight 2.0 \
    --use_wandb
```

**Thay đổi quan trọng**:
- ✅ `lora_r=32` (tăng từ 8): Nhiều capacity hơn 4x để học voice patterns
- ✅ `lora_alpha=64` (tăng từ 16): Tỷ lệ scaling tương ứng
- ✅ Target modules: Bao gồm cả MLP layers (gate_proj, up_proj, down_proj)
- ✅ `text_weight=0.05`: Giảm focus vào text (học nhanh)  
- ✅ `speech_weight=2.0`: Tăng focus vào voice quality
- ✅ `lr=5e-4`: Learning rate cao hơn cho LoRA (từ 1e-4)
- ✅ `epochs=20`: Train lâu hơn để học tốt voice

**Kết quả mong đợi**: Voice adaptation từ 3/10 → 7-9/10

**Các tính năng nâng cao đã bật mặc định:**
- ✅ **Safe Z-Loss**: Ổn định training, tránh NaN.
- ✅ **Dynamic Batching**: Gom nhóm audio cùng độ dài -> Train nhanh hơn.
- ✅ **Gradient Checkpointing**: Tiết kiệm 30% VRAM.
- ✅ **Best Model Saving**: Tự động lưu model tốt nhất theo loss giọng nói.

---

## 5. Monitoring (WandB)

Script hỗ trợ Weights & Biases để theo dõi biểu đồ loss trực quan.

Thêm `--use_wandb` vào lệnh training.
- Project mặc định: `vietnamese-tts`
- Biểu đồ quan trọng cần theo dõi:
  - `loss/loss_speech`: Loss phần sinh giọng nói (quan trọng nhất).
  - `loss/loss_text`: Loss phần dự đoán text (thường giảm nhanh).
  - `loss/total_loss`: Tổng loss.

---

## 6. Tiếp tục Training (Resume)

Nếu quá trình training bị ngắt, chạy lại lệnh cũ. Script sẽ tự động tìm checkpoint gần nhất trong `output_dir` để tiếp tục.

---

## 7. Xử lý lỗi thường gặp

**Lỗi: `CUDA out of memory`**
- Giảm `--batch_size` (vd: từ 8 xuống 4 hoặc 2).
- Bật `--gradient_checkpointing` (mặc định đã bật).
- Dùng LoRA (`--use_lora`) thay vì full finetune.

**Lỗi: `Loss = NaN`**
- Script đã tích hợp `SafeZLoss` và `Logit Clamping` để xử lý việc này.
- Nếu vẫn bị, thử giảm `--lr` xuống (vd: 1e-5).
- Kiểm tra lại dataset xem có audio bị lỗi (quá ngắn < 0.5s hoặc im lặng hoàn toàn) không.
