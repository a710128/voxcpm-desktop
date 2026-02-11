# VoxCPM -> Candle(Rust) 执行方案

## 目标

- 在仓库根目录实现一个 Rust 库，使外部可以直接创建 VoxCPM 实例并进行推理（生成音频波形）。
- 用 Candle 作为张量/算子与推理框架，替代 PyTorch。
- 首先对齐 `VoxCPM/src/voxcpm/core.py` 的高层 API 与 `VoxCPM/src/voxcpm/model/voxcpm.py` 的推理链路；训练相关代码不在本阶段范围。

## 非目标（第一阶段不做）

- 不做 Tauri/GUI 集成。
- 不追求与 Python 端完全一致的性能/数值逐点一致（以“可用 + 误差可控 + 行为一致”为目标）。
- 不实现训练/微调；LoRA 先支持加载与合并/在线注入，训练不做。

## 参考：PyTorch 版本结构（实现对齐对象）

- 高层封装与推理入口：`VoxCPM/src/voxcpm/core.py`
- 核心模型与推理循环：`VoxCPM/src/voxcpm/model/voxcpm.py`
- MiniCPM4 Transformer 与 KV cache：`VoxCPM/src/voxcpm/modules/minicpm4/model.py`，`VoxCPM/src/voxcpm/modules/minicpm4/cache.py`
- Local Encoder：`VoxCPM/src/voxcpm/modules/locenc/local_encoder.py`
- Local DiT + 采样器（UnifiedCFM）：`VoxCPM/src/voxcpm/modules/locdit/local_dit.py`，`VoxCPM/src/voxcpm/modules/locdit/unified_cfm.py`
- AudioVAE：`VoxCPM/src/voxcpm/modules/audiovae/audio_vae.py`
- LoRA：`VoxCPM/src/voxcpm/modules/layers/lora.py`

## Rust 产物形态

在仓库根目录创建一个 Cargo workspace（或单 crate；推荐 workspace 便于拆分与测试）。建议结构：

```
./Cargo.toml
./crates/
  voxcpm/
    Cargo.toml
    src/
      lib.rs
      error.rs
      config.rs
      weights.rs
      tokenizer.rs
      audio.rs
      model/
        mod.rs
        voxcpm.rs
        minicpm4.rs
        cache.rs
        locenc.rs
        locdit.rs
        unified_cfm.rs
        audiovae.rs
        lora.rs
      generation.rs
    examples/
      infer.rs
  voxcpm-cli/            (可选)
    Cargo.toml
    src/main.rs
./tools/
  convert_weights.py     (权重转换脚本，见下文)
```

对外只保证 `crates/voxcpm` 是稳定库接口；CLI 可作为验证工具。

## 对外 API 设计（第一阶段）

目标：使用方式接近 Python 的 `VoxCPM.from_pretrained(...).generate(...)`。

```rust
// crates/voxcpm/src/lib.rs
pub struct VoxCpm { /* 持有模型、tokenizer、采样参数、device 等 */ }

pub struct VoxCpmBuilder { /* 可选：device/precision/threads 等 */ }

pub struct GenerateArgs<'a> {
    pub text: &'a str,
    pub prompt_wav: Option<WavInput<'a>>, // 参考音频（clone/voice prompt）
    pub seed: u64,
    pub max_steps: usize,
    pub guidance_scale: f64,
}

pub enum WavInput<'a> {
    Path(&'a std::path::Path),
    Samples { pcm_f32: &'a [f32], sample_rate: u32 },
}

impl VoxCpm {
    pub fn from_dir(path: impl AsRef<std::path::Path>, builder: VoxCpmBuilder) -> Result<Self>;
    // 可选：feature = "hf" 后支持
    pub fn from_hf(repo: &str, revision: Option<&str>, builder: VoxCpmBuilder) -> Result<Self>;

    pub fn generate(&mut self, args: GenerateArgs<'_>) -> Result<GeneratedAudio>;
}

pub struct GeneratedAudio {
    pub pcm_f32: Vec<f32>,
    pub sample_rate: u32,
}
```

说明：

- `generate` 先做“端到端可用”；`generate_streaming` 作为第二阶段扩展（需要把音频 patch decode 做成流式输出）。
- `VoxCpmBuilder` 中预留：设备（CPU/CUDA/Metal）、dtype（f16/bf16/f32）、是否启用 flash-attn（如果 Candle 后端支持）、线程数等。

## 关键实现拆分（模块对齐）

### 1) 配置与权重加载

- [x] `config.json`：Rust 侧实现 `VoxCpmConfig` JSON 解析与 `dtype` 解析；并支持从常见嵌套位置提取 MiniCPM4 配置（`hidden_size/num_hidden_layers/...`）。
- [x] 权重路径发现：`ModelPaths::discover`（`config.json`/`tokenizer.json`/可选 `model.safetensors`/`audiovae.safetensors`/`lora_weights.safetensors`）。
- [x] safetensors 权重加载 helper：基于 `VarBuilder::from_buffered_safetensors`（默认不使用 `unsafe mmap`）。
- [x] `pytorch_model.bin` / `audiovae.pth`：通过 `tools/convert_weights.py` 预转换为 safetensors（见“权重转换方案”）。

### 2) Tokenizer

- Python 使用 `LlamaTokenizerFast.from_pretrained(path)`；Rust 侧用 `tokenizers` crate 读取同目录 tokenizer 文件（`tokenizer.json`/`tokenizer.model` 等）。
- 实现与 Python 一致的 special tokens / bos/eos 处理，保证文本到 token id 的一致性。

### 3) MiniCPM4 Transformer（基础 LM + residual LM）

对齐 `VoxCPM/src/voxcpm/modules/minicpm4/model.py`：

- RMSNorm
- RoPE（cos/sin cache + apply）
- GQA attention + KV cache（对齐 `cache.py` 的增量解码行为）
- MLP（SiLU gate * up，然后 down）
- `forward`（全序列）与 `forward_step`（单步增量）两套路径

实现策略：

- Candle-NN 里用 `Linear`、`LayerNorm`/自定义 RMSNorm、`Tensor::matmul`、softmax 等拼装。
- KV cache 用 `Tensor` 预分配或 `Vec<Tensor>` 存每层 K/V，支持 append。

### 4) Local Encoder（VoxCPMLocEnc）

对齐 `VoxCPM/src/voxcpm/modules/locenc/local_encoder.py`：

- 输入：音频 patch latent 序列
- 处理：加 learnable special token -> MiniCPMModel（非因果） -> 取 CLS 输出 -> 投影

### 5) Local DiT + UnifiedCFM 采样

对齐 `VoxCPM/src/voxcpm/modules/locdit/local_dit.py` 与 `unified_cfm.py`：

- timestep embedding
- 条件拼接（LM hidden + prefix features）
- Euler steps + CFG（classifier-free guidance）

优先实现“行为一致”的 sampler（步数、噪声调度、CFG 混合方式），后续再做性能优化。

### 6) ScalarQuantizationLayer（FSQ）

对齐 `VoxCPM/src/voxcpm/modules/layers/scalar_quantization_layer.py`：

- 推理时走硬 round-to-grid。

### 7) AudioVAE

对齐 `VoxCPM/src/voxcpm/modules/audiovae/audio_vae.py`：

- Conv1d / ConvTranspose1d 堆叠
- Snake 激活、weight norm 等
- `encode`（波形 -> latent）与 `decode`（latent -> 波形）

注：Candle 侧 1D 卷积与转置卷积是否齐全需要确认；若缺失，计划：

1) 先用等价的 `conv1d` 自实现（im2col + matmul）完成正确性；
2) 后续再替换成更高效实现。

### 8) LoRA

对齐 `VoxCPM/src/voxcpm/modules/layers/lora.py`：

- 第一阶段支持：加载 LoRA 权重并在推理时注入到对应 Linear（W + scale * B@A）。
- 提供两种模式：
  - merge：加载时直接合并到 base 权重（节省推理开销）
  - runtime：不改 base 权重，前向里叠加（便于热插拔）

## 权重转换方案（必须项）

Python 侧模型目录通常包含：`config.json`、tokenizer 文件、`model.safetensors` 或 `pytorch_model.bin`、`audiovae.pth`、可选 `lora_weights.*`。

Rust 侧计划统一读：

- `model.safetensors`
- `audiovae.safetensors`（由转换脚本生成）
- `lora_weights.safetensors`（可选）

实现 `tools/convert_weights.py`：

- 输入：VoxCPM 的模型目录
- 输出：
  - 若仅有 `pytorch_model.bin`：导出为 `model.safetensors`
  - 从 `audiovae.pth` 导出 `audiovae.safetensors`
- 关键点：
  - 保持 key 命名与 Rust 侧模块参数名严格一致（或提供一层 key remap 表）。
  - 处理 Python 中可能出现的 `state_dict` 包装与 `torch.compile` 的 `_orig_mod.` 前缀。

- [x] 已实现：`tools/convert_weights.py`（支持常见 wrapper（`state_dict/model/net/module`）并 strip `_orig_mod.`/`module.` 前缀）。

该转换脚本也将用于“对齐测试”（同一份权重被 Python/Rust 读取）。

## 分阶段里程碑

### Milestone 0：仓库脚手架

- [x] 建立 Cargo workspace + `crates/voxcpm`，编译通过。（已落地：`Cargo.toml`，`crates/voxcpm/`）
- [x] 定义错误类型、配置结构体、权重加载框架（先不实现所有层）。
  - 已落地：`crates/voxcpm/src/error.rs`，`crates/voxcpm/src/config.rs`，`crates/voxcpm/src/weights.rs`
- [x] 添加 `examples/infer.rs`（先输出占位错误，确保 API 形态稳定）。
  - 已落地：`crates/voxcpm/examples/infer.rs`（加载 config/tokenizer；可选验证 MiniCPM4 权重可加载；端到端推理仍为占位）。

补充（已提前完成，原计划在后续里程碑中用到）：

- [x] Tokenizer 基础加载（`tokenizer.json`）与文本编码占位 API。
  - 已落地：`crates/voxcpm/src/tokenizer.rs`
- [x] dtype 策略：默认 BF16；CPU 端 fallback 到 FP32。
  - 已落地：`crates/voxcpm/src/lib.rs`

### Milestone 1：MiniCPM4 Transformer 最小可用

- [x] 实现 RMSNorm/RoPE/Attention/MLP/KV cache。
  - 已落地：`crates/voxcpm/src/model/minicpm4.rs`，`crates/voxcpm/src/model/cache.rs`
- [x] 加载一个小随机权重（或极小 config）跑通 `forward_step`。
  - 已落地：`crates/voxcpm/src/model/minicpm4.rs`（单测）
- [x] MiniCPM4 loader helper：`VoxCpm::load_minicpm4_with_prefix`（支持可选 safetensors key 前缀；用于验证 key/shape 对齐）。
- [ ] 单元测试：
  - [ ] RoPE apply 形状/数值一致性（与 Python 对照导出小张量）
  - [x] KV cache append/读取行为（已通过 `forward_step` smoke test 覆盖基本行为；后续补更细粒度测试）

### Milestone 2：AudioVAE encode/decode 可用

- [x] 实现 AudioVAE decode（对齐 PyTorch 的 causal padding/transpose trim、MRF 残差单元的 dilation=1/3/9 顺序执行、末端 Tanh）并提供 Rust 单测。
  - 已落地：`crates/voxcpm/src/model/audiovae.rs`
- [x] 增加 AudioVAE 可选加载入口（若存在 `audiovae.safetensors` 则可构建）。
  - 已落地：`crates/voxcpm/src/lib.rs`（`load_audiovae_with_prefix`）
- [x] 实现 encode（用于 prompt wav）：输入 `[B,T]`/`[B,1,T]`，按 `hop_length=prod(encoder_rates)` 右侧 padding；推理返回 `mu`（与 PyTorch 一致）。
  - 已落地：`crates/voxcpm/src/model/audiovae.rs`
- [ ] 与 Python 对齐测试（固定权重 + 小输入）。
- [x] 把 `GenerateArgs.prompt_wav` 的读取/重采样/mono 处理链路接入到 Rust 端的 `AudioVae::encode`（对齐 `VoxCPM/src/voxcpm/model/voxcpm.py` 的 prompt 预处理）。
  - 已落地：`crates/voxcpm/src/audio.rs`（WAV 读取 + mono mixdown + rubato 重采样）
  - 已落地：`crates/voxcpm/src/lib.rs`（`VoxCpm::encode_prompt_wav` helper）
  - 依赖：`crates/voxcpm/Cargo.toml`（`hound`/`rubato`）

### Milestone 3：Local Encoder + Local DiT + UnifiedCFM

- [x] 实现 LocEnc、LocDiT、UnifiedCFM sampler。
  - 已落地：`crates/voxcpm/src/model/locenc.rs`，`crates/voxcpm/src/model/locdit.rs`，`crates/voxcpm/src/model/unified_cfm.rs`
  - 模块导出：`crates/voxcpm/src/model/mod.rs`
  - 配置解析补齐：`crates/voxcpm/src/config.rs`（`encoder_config`/`dit_config`/`cfm_config` + 派生 LocEnc/LocDiT MiniCPM4 配置）
  - 依赖：`crates/voxcpm/Cargo.toml`（`rand`/`rand_chacha`/`rand_distr`，用于 sampler seed 可复现）
- [x] 单元测试：Euler steps、CFG 混合、seed 可复现。
  - 已落地：`crates/voxcpm/src/model/unified_cfm.rs`（tests）

### Milestone 4：端到端 VoxCPM 推理（非流式）

- [x] 复刻 `VoxCPMModel._inference` 的主循环（非流式）：
  - 文本 token -> base_lm
  - prompt 音频 -> VAE -> LocEnc
  - 迭代生成 patch latent（UnifiedCFM + LocDiT）-> VAE decode -> 输出 wav
  - 已落地：`crates/voxcpm/src/lib.rs`（`VoxCpm::generate`）
- [x] 补齐 FSQ（ScalarQuantizationLayer）推理路径。
  - 已落地：`crates/voxcpm/src/model/fsq.rs`
- [x] 端到端示例：加载转换后的权重，对一段短文本生成 wav 文件。
  - 已落地：`crates/voxcpm/examples/infer.rs`（输出 `./out.wav`）

补充（本地对齐测试规划；不要求进仓库）

- [ ] 本地数值对齐测试（PyTorch <-> Rust），覆盖模型中主要模块的 forward 精度误差。
  - 目标：用“真实权重”做回归，快速定位某个模块的数值偏差/shape/key 对齐问题。
  - 产物不入 git：fixture 只在本地生成（例如 `./parity_out/`），测试默认 `#[ignore]`。
  - 基本约束（降低 flaky）：
    - 统一 CPU + FP32 生成与比较（避免 GPU kernel/混合精度差异）。
    - PyTorch 侧 `eval()` + `inference_mode()` + 固定 seed；建议 `torch.set_num_threads(1)`。
    - Attention 参考输出避免调用 PyTorch SDPA/flash（用显式 matmul+mask+softmax 实现，和 Rust 逻辑一致）。
    - UnifiedCFM 不对齐 `sample()` 随机起点；对齐 `solve_euler(x0, t_span, ...)`，其中 `x0` 固定写入 fixture。
    - FSQ/round 类模块输入避开临界值（例如 0.5/scale 附近），减少 round 细节差异。
  - 建议工具/测试形态：
    - Python 导出脚本：`tools/export_parity.py`
      - 输入：`--model_dir <ckpt_dir>`（真实权重目录，含 `model.safetensors` / `audiovae.safetensors` / `config.json` / tokenizer）。
      - 输出：`--out_dir ./parity_out`
      - 每个模块/用例写出：`io.safetensors`（inputs + expected outputs）与 `meta.json`（rtol/atol/说明）。
    - Rust 对齐测试（默认忽略）：`crates/voxcpm/tests/parity_real_weights.rs`
      - 通过环境变量指定路径：
        - `VOXCPM_MODEL_DIR=/abs/path/to/ckpt_dir`
        - `VOXCPM_PARITY_DIR=/abs/path/to/parity_out`
        - 可选：`VOXCPM_WEIGHTS_PREFIX=...`（当 safetensors key 具备统一前缀）
      - 运行：`cargo test -p voxcpm -- --ignored`
  - 覆盖模块建议（先小后大）：
    - MiniCPM4：RMSNorm、RoPE、MLP、Attention（显式实现参考）、DecoderLayer、KV cache（prefill vs step）。
    - LocEnc、LocDiT（含 timestep embedding）、UnifiedCFM.solve_euler。
    - FSQ、AudioVAE encode/decode（关闭 noise block）。

### Milestone 5：LoRA 支持 + 体验补齐

- 支持从文件/目录加载 `lora_weights.safetensors` 并 merge/runtime。
- 可选：实现简单 CLI（与 `VoxCPM/src/voxcpm/cli.py` 功能对齐一部分）用于自测。

### Milestone 6：流式生成与优化（可选）

- `generate_streaming`：逐 patch decode、边生成边输出音频 buffer。
- 性能：
  - attention 优化（如 Candle 后端支持）
  - 更高效 conv1d
  - dtype 改成 f16/bf16

## 验证策略（保证可用与可回归）

### 1) 单元测试（Rust）

- RoPE、RMSNorm、Attention（含 GQA）、KV cache、Euler sampler、FSQ round 等。

### 2) 对齐测试（Python <-> Rust）

- 通过 `tools/convert_weights.py` 导出一组“小输入 + 期望输出”（或中间层输出）到 `.npz`/`.safetensors`。
- Rust 侧读取并比较：允许一定误差（f32/f16 差异），核心关注形状与趋势一致。

### 3) 端到端验收

- `examples/infer.rs`：
  - 输入 text + 可选 prompt wav
  - 输出 `out.wav`
- 验收指标：
  - 程序稳定运行
  - 输出 wav 采样率正确、长度合理、非全零/非 NaN
  - 固定 seed 可复现

## 关键风险与处理

- Candle 算子覆盖：若缺少 conv1d/transpose conv 或某些 broadcast/softmax 组合，先用可行但慢的实现保证正确性，再迭代优化。
- 权重 key 对齐：Python 模型 key 命名复杂（含 LoRA/torch.compile 前缀）。优先用“转换脚本 + 显式 key remap”把复杂度从 Rust 侧移走。
- 数值差异：先用 f32 端到端跑通，再逐步切到 f16/bf16。
- 依赖体积与编译时间：把 CLI、HF 下载等做成 feature，默认只包含纯本地推理所需。

## 需要你确认的 2 个决策（开始实现前）

1) Rust 库的 crate 名称：`voxcpm`（已确认）
2) 权重分发策略：先运行 `tools/convert_weights.py` 生成 safetensors，Rust 端只读 safetensors（已确认）
