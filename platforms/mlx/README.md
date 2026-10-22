# MLX platform

在自己的 Apple Silicon Mac 上，透過 MLX / MLX-LM 學習 LLM inference 的效能瓶頸。

這不是一個以「做出聊天應用程式」為目標的專案，而是一個可以重複測量、畫出效能曲線，並解釋曲線為什麼長這樣的實驗室。

## 目前的研究問題

第一階段先回答一個問題：

> 當 context length 改變時，LLM 的 prompt processing、token generation 與記憶體使用量如何改變？

我們先建立可靠的 BF16 baseline，再討論 prefill、decode、KV cache、memory bandwidth，以及之後是否值得寫 custom Metal kernel。

## 實驗原則

- 平台：Apple Silicon Mac，使用 MLX 的 unified memory。
- baseline：BF16、小型模型、batch size = 1。
- context length：128、512、1024、2048 tokens。
- generation length：每次固定 128 tokens。
- 每個設定先 warm up，再進行多次 trial，保留原始結果。
- 一次只改一個變因，先測量再優化。

目前刻意不做：

- custom Metal kernel
- quantization
- CUDA
- DeepSeek 或大型模型
- 自己實作完整 attention

這些主題不是不重要，而是要等 baseline 與結果解讀穩定後再加入，避免不知道效能差異究竟來自哪裡。

## Milestones

### M1：可重現的 LLM baseline

使用 MLX-LM 測量四種 context length，記錄：

- prompt throughput（prefill，prompt tok/s）
- generation throughput（decode，generation tok/s）
- peak memory
- Mac 型號、記憶體、MLX / MLX-LM 版本、模型與精度

官方也提供 `mlx_lm.benchmark`，可以先用它取得標準化的 benchmark 結果：

```bash
python -m mlx_lm.benchmark --model <model> -p 2048 -g 128
```

本 repo 的 `benchmark/inference.py` 則用固定的四組 context length 執行多次 trial，並把原始資料存成 JSON，方便後續分析。

### M2：解釋 baseline

完成兩張圖：

1. context length vs. prompt tok/s
2. context length vs. generation tok/s

並回答：

- 為什麼 prompt throughput 與 generation throughput 差很多？
- context 變長時，哪一個下降比較明顯？
- 記憶體如何成長？其中哪些部分與 KV cache 有關？

### M3：RMSNorm microbenchmark

不先改完整 LLM，只測一個算子，並比較：

```text
naive MLX primitives
        ↓
mx.fast.rms_norm
        ↓
custom Metal RMSNorm
```

先做 correctness，再測 hidden size 1024、2048、4096 等不同 tensor size 的 latency。

### M4：Custom Metal kernel

等 M1–M3 的測量流程與解釋完成後，才使用 `mx.fast.metal_kernel()` 從 Python JIT compile 第一個 custom kernel。目標不是一定超越 MLX 的 optimized implementation，而是理解不同 tensor size 下的成本與限制。

## Repository structure

```text
llm-inference-lab/
├── platforms/
│   ├── mlx/
│   │   ├── benchmark/
│   │   │   ├── inference.py   # M1 baseline runner，輸出 JSON
│   │   │   ├── compare.py      # 比較兩組結果（後續完成）
│   │   │   └── results/        # benchmark 結果
│   │   ├── pyproject.toml      # MLX / MLX-LM dependencies
│   │   └── uv.lock
│   ├── cuda/                   # 未來加入 CUDA 平台
│   └── amd/                    # 未來加入 AMD 平台
├── scripts/
│   └── download_model.py       # 下載本地 MLX 模型
├── model/                      # 本地模型，不提交到 Git
├── plots/                      # M2 圖表（規劃中）
├── kernels/                    # M4 Metal kernels（規劃中）
└── README.md
```

平台相關的程式放在 `platforms/<platform>/`。因此目前的 MLX 實驗全部在 `platforms/mlx/`；未來 CUDA 與 AMD 可以各自擁有 dependencies、benchmark runner 與平台特有的 kernel，而不污染其他平台。

## macOS setup

需要 Apple Silicon Mac、Python 3.11+ 與 [uv](https://docs.astral.sh/uv/)。在 repo 根目錄執行：

```bash
cd platforms/mlx
uv sync
```

下載目前選定的 BF16 小型模型：

```bash
cd platforms/mlx
uv run python ../../scripts/download_model.py
```

模型會放在 repo 根目錄的 `model/`，該目錄已排除於 Git 之外。第一次執行前，應先確認這個模型能被目前的 MLX-LM benchmark runner 正常載入；模型若是 multimodal conversion，可能需要改用相容的 text-only MLX-LM 模型。

## 執行 M1

先使用官方 benchmark 熟悉輸出格式：

```bash
cd platforms/mlx
uv run python -m mlx_lm.benchmark --model ../../model -p 2048 -g 128
```

再執行本 repo 的四組 context length runner：

```bash
cd platforms/mlx
uv run python benchmark/inference.py \
  --model-path ../../model \
  --label baseline
```

結果會寫入：

```text
platforms/mlx/benchmark/results/baseline_<timestamp>.json
```

目前 runner 預設為 128 / 512 / 1024 / 2048 context、每組 128 generation tokens、3 次 timed trials。結果中的 `avg` 是分析用的平均值，`trials` 則保留每次原始測量。每次 trial 也會記錄 `peak_memory_gb`、`active_memory_gb` 與 `allocator_cache_memory_gb`，方便區分 MLX 的不同記憶體統計；其中 allocator cache 不是模型語意上的 KV cache。

## 如何閱讀結果

- **Prefill / prompt processing**：模型一次讀入整段 prompt，通常較適合平行運算，因此 prompt tok/s 可能較高。
- **Decode / generation**：模型逐 token 產生輸出，每一步都會依賴前一步，通常更受記憶體搬移與 KV cache 影響。
- **KV cache**：保存已處理 token 的 attention 狀態，能避免每個 decode step 重新計算整段歷史，但 context 越長通常需要更多記憶體。
- **Peak memory**：在 Apple Silicon 上要連同模型權重、暫存 tensor 與 cache 一起觀察，不能只看模型檔案大小。

比較結果時，模型、精度、prompt 長度、generation 長度、batch size 與硬體都要一致；否則 speedup 或 memory ratio 沒有可靠的意義。

## 進度

- [x] 確立 Mac + MLX-LM 的研究範圍
- [x] 建立 M1 baseline runner
- [ ] 驗證模型與 benchmark runner 可正常執行
- [ ] 完成 128 / 512 / 1024 / 2048 的 baseline 結果
- [ ] 將結果整理成 CSV 並繪製 M2 圖表
- [ ] 寫下對 prefill、decode、KV cache 與 memory 的解釋
- [ ] 開始 RMSNorm microbenchmark
- [ ] 撰寫第一個 custom Metal kernel

## References

- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM benchmark documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md)
- [MLX fast primitives](https://ml-explore.github.io/mlx/build/html/python/fast.html)
- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
