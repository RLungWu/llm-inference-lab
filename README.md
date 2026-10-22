# llm-inference-lab

跨平台的 LLM inference 效能實驗專案。

目前先從 Apple Silicon Mac 上的 MLX / MLX-LM 開始，之後預計加入 CUDA 與 AMD 平台。平台相關的程式與依賴放在 `platforms/<platform>/`，讓各平台可以獨立演進。

## Platforms

- [MLX platform](platforms/mlx/README.md)：Apple Silicon Mac 的 baseline 與效能實驗
- `platforms/cuda/`：規劃中
- `platforms/amd/`：規劃中

## Repository structure

```text
llm-inference-lab/
├── platforms/
│   ├── mlx/                    # MLX platform
│   ├── cuda/                   # 未來加入 CUDA 平台
│   └── amd/                    # 未來加入 AMD 平台
├── scripts/                    # 跨平台共用腳本
├── model/                      # 本地模型，不提交到 Git
├── plots/                      # 實驗圖表
├── kernels/                    # 平台特有 kernels
└── README.md
```

詳細的安裝、benchmark 指令與目前進度，請參考 [MLX platform README](platforms/mlx/README.md)。
