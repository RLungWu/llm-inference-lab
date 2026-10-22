# CLAUDE.md

跨平台 LLM inference 效能實驗，目的是學習底層優化。目前在 Apple Silicon Mac 上用 MLX / MLX-LM。
專案背景見 [README.md](README.md) 與 [platforms/mlx/README.md](platforms/mlx/README.md)。

## 協作方式

- 這是學習專案，**程式碼由我自己寫**。Claude 負責給提示、解釋概念、review 我寫的程式。
- 除非我明確要求，不要用工具直接建檔、寫骨架或修改我的程式。
- 給提示時指出檔案與行號，說明「為什麼」，不要直接給完整答案。
- 不確定我要哪種協助時，先問。
- 筆記草稿、文件等，我要求時可以直接寫。

## 目前進度

- [x] M1：可重現的 baseline runner（`platforms/mlx/benchmark/inference.py`）
- [x] M2：解釋 baseline，見 [M2_NOTES.md](platforms/mlx/benchmark/M2_NOTES.md)
- [ ] M3：RMSNorm microbenchmark，起點問題在 M2_NOTES 的「Next：M3」
- [ ] M4：Custom Metal kernel

開始新的 milestone 前，先根據上一階段的 NOTES 出幾題快速複習。換新情境出題，不要重考原題。

## 慣例

- MLX 相關指令在 `platforms/mlx/` 底下用 `uv run` 執行（uv 依目前目錄找 `pyproject.toml` 決定環境）。
- 模型放在 repo 根目錄的 `model/`，不進 git。
- Benchmark 結果存成 JSON 到 `platforms/mlx/benchmark/results/`，用 `--label` 區分；metadata 要記錄所有實驗變因。

## 實驗原則

- 一次只改一個變因，並保留一組重現 baseline 的對照組。
- 跑之前先寫下預測，跑完再對照。
- 結果異常時，先重跑排除雜訊，再推論機制。
- 估算吻合只是假設，要靠讀程式碼或實驗確認。
