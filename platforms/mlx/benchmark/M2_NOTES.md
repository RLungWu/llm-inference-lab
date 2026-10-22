# M2 Baseline Analysis

## Environment

- Model: mlx-community/Qwen3.5-0.8b-bf16
- Chip: Apple M5
- Memory: 24 GB
- macOS: Tahoe 26.5.1
- MLX: 0.32.2
- MLX-LM: 0.31.3
- 設定：BF16、batch size 1、generation 128 tokens、warmup 1 次 + 3 次 trial、`prefill_step_size` 預設 2048

### 模型架構（與記憶體相關的部分）

從 `model/config.json`：

| 參數 | 值 | 備註 |
|---|---|---|
| `num_hidden_layers` | 24 | |
| `layer_types` | 6 × full_attention、18 × linear_attention | `full_attention_interval: 4`，每 4 層 1 層 full attention |
| `num_attention_heads` | 8 | |
| `num_key_value_heads` | 2 | GQA：8 個 query head 共用 2 組 K/V |
| `head_dim` | 256 | full attention 用的。不要跟 linear attention 的 `linear_key_head_dim`（128）搞混 |
| `hidden_size` / `intermediate_size` | 1024 / 3584 | |

MLX-LM 裡 full attention 層用 `KVCache`（隨 token 數成長），linear attention 層用 `ArraysCache`（固定大小，跟 context 長度無關），見 `mlx_lm/models/qwen3_5.py` 的 `make_cache()`。

### 資料來源

- Baseline：`results/baseline_m2_20260903_191307.json`
- Prefill chunk 實驗：`results/prefill_{2048,1024,512,256}_20261021_*.json`

---

## 結論摘要（回答 README 的 M2 問題）

**為什麼 prompt throughput 與 generation throughput 差很多？**
Prefill 一次處理整段 prompt，可以平行計算，硬體吃得滿；decode 每產生一個 token 都要把全部權重從記憶體讀過一次，主要受記憶體頻寬限制（見下方估算）。所以 prompt tok/s（~2600–6500）遠高於 generation tok/s（~76–79）。

**Context 變長時，哪一個下降比較明顯？**
在 128–2048 的範圍內，兩者都沒有明顯下降。Generation 只從 79.0 降到 75.8 tok/s（約 4%）；prompt throughput 反而上升，原因見下方。

**記憶體如何成長？其中哪些部分與 KV cache 有關？**
Peak 從 1.68 GB 增加到 2.66 GB，但 KV cache 只佔其中很小一部分（2048 tokens 約 24 MiB）。**成長主要來自 prefill 時的暫存中間結果**，而且 peak 是由「一次處理多少 token」決定，不是 context 總長度。

---

## 1. Baseline 觀察

| context | prompt tok/s | generation tok/s | peak memory | peak − 權重 |
|---|---|---|---|---|
| 128 | 2613 | 79.0 | 1.678 GB | 0.17 GB |
| 512 | 5026 | 77.9 | 2.094 GB | 0.59 GB |
| 1024 | 5873 | 76.9 | 2.369 GB | 0.86 GB |
| 2048 | 6530 | 75.8 | 2.664 GB | 1.16 GB |

（權重 ≈ `active_memory_gb` = 1.505 GB，說明見第 4 節。）

### Prompt throughput

Prompt throughput 從 2613 上升到 6530 tok/s，約 2.5 倍。

這不代表長 context 的處理成本變低。prompt_tps 是 prompt tokens 除以到第一個 token 的 prompt processing time。短 prompt 比較容易受到固定的 first-token、dispatch 等成本影響；context 變長後，這些固定成本被更多 token 攤平，GPU 也比較能吃滿。

### Generation throughput

Generation throughput 約維持在 75.8–79.0 tok/s，在目前的 context 範圍內沒有看到明顯的 context slowdown。

Decode 時，Q 是目前新生成 token 的 query；K/V 則是目前 token 與過去 token 的 attention state。KV cache 避免重新計算過去 token，但每一步仍需要讀取歷史 cache。

**Memory bandwidth 估算（待確認）**：decode 每產生一個 token，都要讀過一次全部權重（~1.5 GB）。
1.5 GB × 76 tok/s ≈ **114 GB/s**。如果 M5 的記憶體頻寬約 150 GB/s（需查證規格），decode 已經用掉約 3/4 的頻寬。這比「memory bandwidth 是合理假設」更有力的支持，但仍然只是估算，不是量測。相較之下，KV cache 每步只需要讀幾十 MB，可以忽略。

---

## 2. KV cache 理論估算

只有 full attention 層會存 KV cache。每層、每個 token：

```
K、V 兩份 × num_key_value_heads × head_dim × bytes
= 2 × 2 × 256 × 2 (BF16) = 2,048 bytes = 2 KB
```

| 情境 | 每 token | 2048 tokens |
|---|---|---|
| 實際：6 層 full attention + GQA（2 KV heads） | 12 KB | ~24 MiB |
| 假設：24 層全 full attention、沒有 GQA（8 KV heads） | 192 KB | ~384 MiB（~403 MB） |

差 16 倍，來自兩個設計：

- **Hybrid attention**：需要 KV cache 的層數從 24 變 6，省 4 倍
- **GQA**：KV heads 從 8 變 2，省 4 倍

8192 tokens 時實際 KV cache 約 96 MiB，在 24 GB（甚至 8 GB）的機器上仍然很小。

MLX 的 `KVCache` 以 256 tokens 為單位預先配置，實際會比公式略大一點。

---

## 3. 實驗：prefill chunk size

### 問題

Context 從 128 到 2048 時多出的 ~1 GB，有多少是「一次 prefill 整段 prompt」造成的？

### 設計

- **唯一變因**：`prefill_step_size` = 2048 / 1024 / 512 / 256
- **固定**：context 2048、generation 128、BF16、batch 1、warmup + 3 trials
- KV cache 大小在四組之間完全相同（context 都是 2048）

### 預測（跑之前寫下）

- 如果大頭是 prefill 中間結果：step 越小 peak 越低
- KV cache 不受 step size 影響
- Decode 不受影響
- 小 step 時 prompt_tps 可能下降（dispatch 次數多、矩陣小）

### 結果

| step size | peak | peak − 權重 | 比上一組少 | prompt tok/s | generation tok/s |
|---|---|---|---|---|---|
| 2048 | 2.664 GB | 1.16 GB | — | 6389 | 76.4 |
| 1024 | 2.396 GB | 0.89 GB | 0.27 GB | 6149（−4%） | 76.7 |
| 512 | 2.144 GB | 0.64 GB | 0.25 GB | 6099（−5%） | 76.4 |
| 256 | 1.902 GB | 0.40 GB | 0.24 GB | 5501（−14%） | 73.0 * |

\* 256 那組的三次 trial 是 76.6 / 71.4 / 71.0。第一次跟其他組一致，後兩次下降。`prefill_step_size` 理論上不影響 decode，較可能是環境雜訊，尚未重跑確認。

### 結論

1. **記憶體成長主要不是 KV cache。** Context 固定 2048、KV cache 一樣大，只改切塊方式，peak 就從 2.66 降到 1.90 GB。這 0.76 GB 的差距來自 prefill 的暫存中間結果。
2. **Peak 由一次處理的 token 數決定，不是 context 總長度。**
   - 切成 1024 的 peak（2.396 GB）≈ baseline ctx=1024 一次送完（2.369 GB），差 27 MB
   - 切成 512 的 peak（2.144 GB）≈ baseline ctx=512 一次送完（2.094 GB），差 50 MB
3. **記憶體與速度的取捨。** 2048 → 512 省 0.52 GB、prompt 只慢 ~5%；再降到 256 只多省 0.24 GB，卻慢 ~14%。這也是 MLX-LM 預設值不設得很小的原因。

### 為什麼相信這個結論

- **單一變因**：只改 chunk 大小，context 與 KV cache 完全固定
- **對照組重現 baseline**：step=2048 的 peak 與 baseline 完全相同（2.664 GB），證明參數有傳到、結果可以比較
- **Decode 不變**：作為 sanity check，證明只有 prefill 被影響
- **跨實驗一致**：切塊後的 peak 與 baseline 中相同大小的 context 接近
- **Peak 有決定性**：同一組三次 trial 的 peak 幾乎完全相同，組間差異不是雜訊

### 切塊後多出來的記憶體（推測）

「ctx=2048 切成 1024」比「ctx=1024 一次送完」多 27 MB，可能來自：

- 多存了 1024 tokens 的 KV cache：1024 × 12 KB ≈ 12 MB
- 第二塊的 1024 個 query 要看全部 2048 個 key：如果 attention score 有完整算出來，多 8 × 1024 × 1024 × 2 bytes ≈ 16 MB

12 + 16 ≈ 28 MB，跟觀察值吻合。但用同樣方法估 512 的情況：KV 多 1536 tokens ≈ 18 MB、attention 多 ≈ 12.6 MB，合計 ≈ 31 MB，觀察值是 50 MB，對不太上。**估算吻合只是假設，不是證明。**

切塊只能限制「一次處理多少 query」，限制不了「要看多少 key」。context 變長時，attention score 是 chunk × T，T 那一項仍會隨 context 成長。

---

## 4. 怎麼讀記憶體數字

- **`peak_memory_gb`**：整次執行（prefill + decode）過程中的最高點。每次 trial 前用 `mx.reset_peak_memory()` 重設，所以每次 trial 是獨立量測。
- **`active_memory_gb`**：在 `stream_generate` 結束**之後**才讀的快照。那時 KV cache 與中間結果都已釋放，只剩權重。驗證：1.505 GB ÷ 2 bytes ≈ 7.5 億參數，跟 0.8B 模型吻合。
- **`allocator_cache_memory_gb`**：MLX allocator 保留以便重用的記憶體，不是模型語意上的 KV cache。

要看過程中的暫存記憶體，只能看 peak；active 看不到。

---

## 5. 待解問題

1. **每次 chunk 減半，peak 都降固定 ~0.25 GB。** 如果記憶體跟 chunk 大小成正比，降幅應該是 0.58 → 0.29 → 0.15 GB，實際卻幾乎不變。只有四個點，還不足以判斷是什麼曲線。可能的方向（都是假設，待驗證）：
   - 18 層 linear attention 在 prefill 時的中間結果
   - MLX lazy evaluation 下同時存活的中間結果數量
   - Allocator 的配置粒度

   需要 kernel 層級的工具（例如 Xcode Instruments 的 Metal trace）或逐層量測，留到 M3/M4 之後再處理。
2. **256 那組的 decode 下降**：重跑確認是否為雜訊。
3. **512 切塊多出的 50 MB**：為什麼比估算的 31 MB 多？
4. **M5 的記憶體頻寬規格**：查證後更新第 1 節的 bandwidth 估算。

---

## 6. Hypotheses

1. **（未驗證）** 如果 generation 主要受 memory bandwidth 限制，增加 batch size 可能提高總體 throughput，但不會線性增加。

2. **（改寫）** Decode 每產生一個 token，是 1 個 query 去看 N 個 key，所以 full attention 層的 decode 成本是 **O(N)**；linear attention 層每個 token 的成本是**固定的**。
   - 實驗：固定 chunk 512，context 從 1024 增加到 8192，量 decode 每 token 的延遲。
   - 預測：延遲隨 context 線性增加，但斜率只來自那 6 層 full attention，所以很平緩。
   - 對照組：找一顆差不多大、每層都是 full attention 的模型（例如 Qwen3-0.6B，需確認架構）跑同樣實驗，預期斜率明顯較陡。

3. **（未驗證）** 如果 prompt_tps 上升主要來自固定 first-token 成本被攤平，在相同 context 下改變 generation length，prompt_tps 應該大致不變。

---

## 7. 方法論筆記

- **先寫預測，再跑實驗。** 這次預測 256 會趨近下限，結果被推翻，反而暴露了「每次固定降 0.25 GB」這個問題。如果跑完才解釋，很容易只說「chunk 變小所以記憶體變少」，忽略更深層的規律。
- **結果異常時，先懷疑雜訊，再推論機制。** 先重跑確認能否重現。
- **數字對得上不等於找到原因。** 例如 prompt 的 logits（2047 × 248,320 vocab × 2 bytes ≈ 1 GB）恰好跟記憶體增量差不多，但讀程式碼發現 MLX-LM prefill 只 eval cache state，logits 根本沒被算出來（lazy evaluation）。
- **四組結果幾乎一樣時，先懷疑參數沒傳到，而不是參數沒效果。** 不會報錯的 bug 最危險。
- **計算完要交叉檢查。** 用兩種方法算同一個量，對不上就代表有參數代錯。

---

## Next：M3（RMSNorm microbenchmark）

M2 的發現帶到 M3 的問題：

1. RMSNorm 在 prefill 和 decode 時，輸入的形狀分別是什麼？（hidden size 1024，想想「一次處理多少 token」。）
2. README 的 M3 計畫只改 hidden size（1024 / 2048 / 4096）。根據上一題，是否也要改「token 數」這個維度？
3. RMSNorm 每個元素只做幾個運算，但每個元素都要讀進來、寫回去。它比較受計算速度限制，還是受記憶體頻寬限制？應該用 FLOPS 還是 GB/s 衡量？
4. 單一算子可能只要幾微秒。MLX 的 lazy evaluation 會讓計時出什麼問題？怎麼確保量到的是真正的計算時間？
