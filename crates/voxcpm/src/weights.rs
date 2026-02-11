use crate::{Result, VoxCpmError};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model_dir: PathBuf,
    pub config_json: PathBuf,
    pub tokenizer_json: PathBuf,
    pub model_safetensors: Option<PathBuf>,
    pub audiovae_safetensors: Option<PathBuf>,
    pub lora_safetensors: Option<PathBuf>,
}

impl ModelPaths {
    pub fn discover(model_dir: &Path) -> Result<Self> {
        let model_dir = model_dir.to_path_buf();

        let config_json = model_dir.join("config.json");
        if !config_json.is_file() {
            return Err(VoxCpmError::InvalidArg(format!(
                "missing required file: {}",
                config_json.display()
            )));
        }

        // First milestone: require tokenizer.json for deterministic loading.
        let tokenizer_json = model_dir.join("tokenizer.json");
        if !tokenizer_json.is_file() {
            return Err(VoxCpmError::InvalidArg(format!(
                "missing required file: {} (tokenizer.json)",
                tokenizer_json.display()
            )));
        }

        let model_safetensors = {
            let p = model_dir.join("model.safetensors");
            p.is_file().then_some(p)
        };

        // Produced by tools/convert_weights.py (planned).
        let audiovae_safetensors = {
            let p = model_dir.join("audiovae.safetensors");
            p.is_file().then_some(p)
        };

        let lora_safetensors = {
            let p = model_dir.join("lora_weights.safetensors");
            p.is_file().then_some(p)
        };

        Ok(Self {
            model_dir,
            config_json,
            tokenizer_json,
            model_safetensors,
            audiovae_safetensors,
            lora_safetensors,
        })
    }

    fn vb_from_safetensors(
        path: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        if !path.is_file() {
            return Err(VoxCpmError::InvalidArg(format!(
                "missing weights file: {}",
                path.display()
            )));
        }
        // Prefer the safe buffered loader (no unsafe mmap).
        let bytes = std::fs::read(path)?;
        Ok(VarBuilder::<'static>::from_buffered_safetensors(
            bytes, dtype, device,
        )?)
    }

    /// Create a `VarBuilder` for the main model weights (e.g. MiniCPM4 / LocEnc / LocDiT).
    pub fn model_var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
        let p = self.model_safetensors.as_ref().ok_or_else(|| {
            VoxCpmError::InvalidArg("missing model.safetensors in model directory".into())
        })?;
        Self::vb_from_safetensors(p, dtype, device)
    }

    /// Create a `VarBuilder` for AudioVAE weights, if present.
    pub fn audiovae_var_builder(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<Option<VarBuilder<'static>>> {
        let Some(p) = self.audiovae_safetensors.as_ref() else {
            return Ok(None);
        };
        Ok(Some(Self::vb_from_safetensors(p, dtype, device)?))
    }

    /// Create a `VarBuilder` for LoRA weights, if present.
    pub fn lora_var_builder(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<Option<VarBuilder<'static>>> {
        let Some(p) = self.lora_safetensors.as_ref() else {
            return Ok(None);
        };
        Ok(Some(Self::vb_from_safetensors(p, dtype, device)?))
    }
}
