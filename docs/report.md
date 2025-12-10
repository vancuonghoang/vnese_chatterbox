# Báo cáo Đánh giá Chất lượng - Dự án Viterbox TTS

**Người đánh giá:** QA Lead (10+ năm kinh nghiệm AI/TTS)
**Ngày đánh giá:** 2025-12-10
**Phiên bản:** Đánh giá dựa trên trạng thái mã nguồn tại thời điểm hiện tại.

---

## 1. Tổng quan dự án

Dự án **Viterbox** là một hệ thống Text-to-Speech (TTS) tiếng Việt, được fine-tune từ mô hình Chatterbox của Resemble AI.

- **Mục tiêu:** Cung cấp giọng đọc tiếng Việt tự nhiên, chất lượng cao với khả năng voice cloning "zero-shot".
- **Kiến trúc:** Sử dụng mô hình T3 (Text-to-Token), S3Gen (Vocoder), và Voice Encoder.
- **Giao diện:** Cung cấp nhiều phương thức tương tác: Web UI (Gradio), Python API, và Command Line (CLI).
- **Luồng huấn luyện:** Hỗ trợ fine-tuning (full-model và LoRA) với quy trình tiền xử lý dữ liệu (pre-computed) để tăng tốc độ.

---

## 2. Đánh giá chung

Dự án được xây dựng tốt, có cấu trúc rõ ràng và tài liệu hướng dẫn người dùng (`README.md`) rất chi tiết, thân thiện. Tuy nhiên, dưới góc độ của một sản phẩm cần sự ổn định và khả năng bảo trì cao, dự án vẫn còn nhiều điểm cần cải thiện, đặc biệt ở khâu đảm bảo chất lượng và sự "bất biến" (robustness) của cả luồng inference và training.

| Hạng mục | Điểm mạnh | Điểm yếu / Rủi ro |
|---|---|---|
| **Tính năng** | Hỗ trợ voice cloning, xử lý văn bản dài, đa dạng giao diện. | Thiếu cơ chế xử lý lỗi đầu vào triệt để. |
| **Mã nguồn** | Cấu trúc module hóa tốt (`viterbox/models`). | Thiếu hoàn toàn bộ kiểm thử tự động (automated tests). |
| **Huấn luyện** | Quy trình pre-computed và LoRA giúp training hiệu quả. | Luồng training còn nhiều "happy path", dễ lỗi với dữ liệu thực tế. |
| **Bảo trì** | `README.md` tốt, có `pyproject.toml`. | `requirements.txt` không pin version, gây rủi ro về môi trường. |
| **Tài liệu** | `README.md` xuất sắc. `TRAINING_GUIDE.md` khá tốt. | `TRAINING_GUIDE.md` thiếu các bước quan trọng (validation, merge LoRA). |

---

## 3. Phân tích chi tiết và Lỗi tiềm ẩn

### 3.1. Luồng Inference (API, Web UI, CLI)

Luồng inference hoạt động tốt trên các kịch bản cơ bản. Tuy nhiên, các lỗi tiềm ẩn chủ yếu đến từ việc xử lý đầu vào.

- **[BUG-INFERENCE-01] Thiếu Input Validation:**
  - **Vấn đề:** `app.py` và `inference.py` không có các bước kiểm tra đầu vào nghiêm ngặt.
  - **Rủi ro:**
    - Người dùng upload file audio không phải định dạng WAV (ví dụ: MP3, M4A) cho `audio_prompt` có thể gây crash.
    - Audio mẫu quá ngắn (<1s) hoặc quá dài (>30s) có thể cho kết quả voice clone kém chất lượng hoặc gây lỗi OOM (Out of Memory).
    - Văn bản đầu vào chứa ký tự đặc biệt, emoji, hoặc không phải tiếng Việt/Anh có thể tạo ra âm thanh không mong muốn.
    - Các tham số như `temperature`, `cfg_weight` nếu nhận giá trị ngoài khoảng cho phép có thể gây lỗi.
  - **Mức độ:** Trung bình.

- **[BUG-INFERENCE-02] Xử lý lỗi không thân thiện:**
  - **Vấn đề:** Khi xảy ra lỗi (ví dụ: OOM trên GPU), ứng dụng có thể bị crash hoàn toàn thay vì hiển thị một thông báo lỗi thân thiện cho người dùng trên giao diện Gradio.
  - **Mức độ:** Thấp.

### 3.2. Luồng Huấn luyện (Training Workflow)

`TRAINING_GUIDE.md` mô tả một quy trình huấn luyện hiện đại. Tuy nhiên, quy trình này được thiết kế theo "happy path" và bỏ qua nhiều bước quan trọng trong một pipeline MLOps thực thụ.

- **[BUG-TRAIN-01] Thiếu bước Data Validation trong `preprocess_dataset.py`:**
  - **Vấn đề:** Script giả định dữ liệu đầu vào (audio và text) đã sạch.
  - **Rủi ro:**
    - Audio bị lỗi (file rỗng, corrupted, định dạng sai) sẽ gây crash pipeline.
    - Audio ở định dạng stereo thay vì mono có thể được xử lý sai cách (chỉ lấy 1 kênh) mà không có cảnh báo, dẫn đến lãng phí tài nguyên và model học sai.
    - Transcript chứa lỗi (ký tự lạ, sai encoding) sẽ ảnh hưởng trực tiếp đến chất lượng tokenizer và model.
  - **Mức độ:** Cao. Đây là rủi ro lớn nhất trong luồng training.

- **[BUG-TRAIN-02] Quy trình Resume Training còn sơ sài:**
  - **Vấn đề:** Hướng dẫn đề cập việc "chạy lại lệnh cũ" để resume. Cơ chế này có thể không đáng tin cậy.
  - **Rủi ro:** Script có thể không load lại đúng trạng thái của optimizer, learning rate scheduler, và số epoch đã chạy, dẫn đến quá trình training không được tiếp tục một cách chính xác.
  - **Mức độ:** Trung bình.

- **[INCOMPLETENESS-TRAIN-03] Thiếu hướng dẫn Merge LoRA:**
  - **Vấn đề:** `TRAINING_GUIDE.md` hướng dẫn cách train LoRA nhưng không có bước cuối cùng: làm thế nào để **merge các trọng số LoRA** vào model gốc để triển khai inference. Người dùng sau khi train xong sẽ không biết cách sử dụng artifact đã tạo ra.
  - **Mức độ:** Cao. Khiến cho luồng LoRA không hoàn chỉnh.

- **[OPTIMIZATION-TRAIN-04] Pipeline Pre-processing không tối ưu cho bộ dữ liệu cực lớn:**
  - **Vấn đề:** `preprocess_dataset.py` xử lý toàn bộ dataset trong một lần chạy.
  - **Rủi ro:** Với bộ dữ liệu hàng nghìn giờ, quá trình này sẽ rất tốn thời gian, tốn bộ nhớ và nếu thất bại giữa chừng sẽ phải chạy lại từ đầu. Các hệ thống lớn thường sử dụng on-the-fly processing hoặc các data loader được tối ưu hơn (ví dụ: `torchdata`, `datasets`).
  - **Mức độ:** Thấp (chỉ ảnh hưởng khi scale lên).

### 3.3. Cấu trúc Code và Bảo trì

- **[RISK-MAINTAIN-01] Không Pin Dependencies:**
  - **Vấn đề:** File `requirements.txt` và `requirements-train.txt` liệt kê các thư viện nhưng không "pin" phiên bản cụ thể (ví dụ: `torch==2.1.0`).
  - **Rủi ro:** Bất kỳ ai cài đặt dự án ở một thời điểm khác trong tương lai có thể nhận được phiên bản thư viện khác, dẫn đến lỗi không tương thích, kết quả tái tạo thất bại (non-reproducible builds). Đây là một rủi ro nghiêm trọng cho các dự án AI.
  - **Mức độ:** Cao.

- **[RISK-MAINTAIN-02] Thiếu hoàn toàn Kiểm thử tự động (Automated Testing):**
  - **Vấn đề:** Không có thư mục `tests/` và không có bất kỳ unit test, integration test nào.
  - **Rủi ro:** Bất kỳ thay đổi nào trong mã nguồn (ví dụ: refactor một hàm trong `viterbox/tts.py`, sửa lỗi trong `train/trainer.py`) đều có nguy cơ làm hỏng các tính năng khác một cách âm thầm. Điều này làm cho việc bảo trì và phát triển về lâu dài trở nên cực kỳ rủi ro và tốn kém.
  - **Mức độ:** Rất cao.

- **[RISK-MAINTAIN-03] Cấu hình phân tán:**
  - **Vấn đề:** Cấu hình được quản lý qua các đối số dòng lệnh (CLI arguments). Khi số lượng tham số tăng lên, việc quản lý trở nên phức tạp.
  - **Rủi ro:** Khó theo dõi và tái tạo các lần chạy thí nghiệm với bộ tham số nào.
  - **Mức độ:** Thấp.

---

## 4. Đề xuất & Hướng cải thiện

1.  **Thêm Unit Tests và Integration Tests:**
    - **Hành động:** Tạo thư mục `tests/`.
      - Viết unit test cho các hàm core trong `viterbox/tts.py` (ví dụ: `generate`, `preprocess_text`).
      - Viết integration test cho luồng inference từ đầu đến cuối (CLI và API).
      - Viết test cho luồng training để đảm bảo nó chạy qua một vài steps mà không bị lỗi.
    - **Lợi ích:** Đảm bảo sự ổn định khi thay đổi code, tăng độ tin cậy của dự án.

2.  **Sử dụng Pinned Dependencies:**
    - **Hành động:** Chạy `pip freeze > requirements.lock.txt` và commit file này. Hướng dẫn người dùng cài đặt bằng `pip install -r requirements.lock.txt`.
    - **Lợi ích:** Đảm bảo môi trường có thể tái tạo 100%, loại bỏ rủi ro từ các thư viện phụ thuộc.

3.  **Hoàn thiện luồng Training:**
    - **Hành động:**
      - **Data Validation:** Thêm một bước vào `preprocess_dataset.py` để kiểm tra (và có thể tự động sửa) các file audio lỗi, chuyển đổi sang mono, và làm sạch text. Báo cáo các file không hợp lệ.
      - **Merge LoRA:** Cung cấp một script `merge_lora_weights.py` và hướng dẫn trong `TRAINING_GUIDE.md`.
      - **Robust Resume:** Cải thiện cơ chế resume để lưu và tải lại cả state của optimizer và scheduler.
    - **Lợi ích:** Giúp luồng training trở nên chuyên nghiệp, đáng tin cậy và hoàn chỉnh.

4.  **Tăng cường Input Validation ở lớp Giao diện:**
    - **Hành động:** Trong `app.py` và `inference.py`, thêm logic để kiểm tra định dạng file, độ dài audio, khoảng giá trị của tham số trước khi truyền vào model.
    - **Lợi ích:** Trải nghiệm người dùng tốt hơn, tránh các lỗi không mong muốn.

5.  **Centralize Configuration:**
    - **Hành động:** Cân nhắc sử dụng các thư viện quản lý config như Hydra hoặc đơn giản là dùng file YAML/JSON để quản lý tất cả tham số training.
    - **Lợi ích:** Dễ dàng quản lý, theo dõi và chia sẻ các cấu hình thí nghiệm.

---

## 5. Kết luận

Viterbox là một dự án TTS tiếng Việt rất hứa hẹn với nền tảng công nghệ tốt và tài liệu thân thiện. Các vấn đề được nêu ở trên chủ yếu thuộc về lĩnh vực Kỹ thuật phần mềm và MLOps, vốn thường bị bỏ qua trong các dự án nghiên cứu.

Bằng cách áp dụng các đề xuất trên—đặc biệt là **thêm kiểm thử tự động** và **quản lý dependencies chặt chẽ**—dự án sẽ tăng cường đáng kể độ tin cậy, khả năng bảo trì và sẵn sàng để phát triển thành một sản phẩm vững chắc.

---

## 6. Fix Log (2025-12-11)

### ✅ Critical Bugs Fixed

#### [BUG-LOSS-01] - NaN Detection Now Raises Exception
- **Problem**: Loss calculator returned `0.0` when NaN/Inf was detected, masking serious training issues.
- **Fix**: Modified `T3LossCalculator` to raise `RuntimeError` with detailed diagnostic message.
- **Evidence**: [loss.py:L253-266](file:///Users/cuonghoang1611/Desktop/WORKSPACES/chatterbox-finetune-vi/viterbox-tts/train/loss.py#L253-L266)
- **Impact**: Training will now crash immediately when NaN occurs, forcing investigation of root cause (high LR, bad data, numerical instability).

```python
# Before: Silently returned 0.0
if torch.isnan(total_loss):
    total_loss = torch.tensor(0.0, ...)

# After: Raises with diagnostic info
if torch.isnan(total_loss):
    raise RuntimeError(f"NaN detected! Causes: gradient explosion, data corruption...")
```

#### [BUG-DATA-01] - Data Loading Errors Now Logged
- **Problem**: `LengthGroupedSampler._compute_lengths()` used bare `except:` that silently ignored all errors.
- **Fix**: Added specific exception handling with logging and summary report.
- **Evidence**: [datasets.py:L220-250](file:///Users/cuonghoang1611/Desktop/WORKSPACES/chatterbox-finetune-vi/viterbox-tts/train/datasets.py#L220-L250)
- **Impact**: Failed samples are now logged with error type. Summary shows `⚠️ X/Y samples failed to load`.

```python
# Now logs: "Failed to load sample 42: I/O error - FileNotFoundError"
# Shows summary: "⚠️ 15/1000 samples failed to load"
```

#### [BUG-TRAIN-01] - Checkpoint Resume Completely Rewritten
- **Problem**: `BestModelCallback` only saved `model.state_dict()`, missing optimizer/scheduler/scaler states.
- **Fix**: 
  1. Removed `BestModelCallback`
  2. Use `load_best_model_at_end=True` in `TrainingArguments`
  3. Added `ResumeVerificationCallback` to log resume details
- **Evidence**: 
  - [trainer.py:L174-241](file:///Users/cuonghoang1611/Desktop/WORKSPACES/chatterbox-finetune-vi/viterbox-tts/train/trainer.py#L174-L241)
  - [run.py:L278](file:///Users/cuonghoang1611/Desktop/WORKSPACES/chatterbox-finetune-vi/viterbox-tts/train/run.py#L278)
- **Impact**: Checkpoint resume now correctly restores optimizer, scheduler, AND model state. Resume logs show step, epoch, and LR.

```python
# Old: Incomplete checkpoint
torch.save(model.t3.state_dict(), "best.pt")  # Missing optimizer!

# New: Use HuggingFace built-in (complete)
TrainingArguments(load_best_model_at_end=True, ...)
```

### 📊 Verification Status

| Bug | Status | Test Required | Priority |
|-----|--------|---------------|----------|
| BUG-LOSS-01 | ✅ Fixed | Test with `--lr 100.0` | HIGH |
| BUG-DATA-01 | ✅ Fixed | Test with corrupted .pt files | HIGH |
| BUG-TRAIN-01 | ✅ Fixed | Test resume (kill + restart) | HIGH |

### 🔄 Next Steps

1. **Testing**: Run verification tests per priority
2. **Data Validation**: Add validators to `preprocess_dataset.py` (Phase 2)
3. **LoRA Merge**: Create merge script (Phase 3)
