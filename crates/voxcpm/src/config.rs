use crate::{Result, VoxCpmError};
use candle_core::DType;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct VoxCpmConfig {
    pub raw: serde_json::Value,
}

impl VoxCpmConfig {
    pub fn from_json_str(s: &str) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(s).map_err(VoxCpmError::Json)?;
        Ok(Self { raw })
    }

    pub fn dtype(&self) -> Option<DType> {
        let s = self.raw.get("dtype")?.as_str()?;
        match s {
            "bfloat16" | "bf16" => Some(DType::BF16),
            "float16" | "fp16" => Some(DType::F16),
            "float32" | "fp32" => Some(DType::F32),
            _ => None,
        }
    }

    /// Extract the MiniCPM4 config used by the text model.
    ///
    /// Different model packages may either store the MiniCPM config at the top-level
    /// or nested under a sub-key (e.g. "text_config"). We try a few common locations.
    pub fn minicpm4(&self) -> Result<crate::model::minicpm4::MiniCpmConfig> {
        use crate::model::minicpm4::MiniCpmConfig;

        fn parse(v: &Value) -> core::result::Result<MiniCpmConfig, serde_json::Error> {
            serde_json::from_value::<MiniCpmConfig>(v.clone())
        }

        // 1) Directly at the top-level.
        if self.raw.get("hidden_size").is_some() {
            return Ok(parse(&self.raw).map_err(VoxCpmError::Json)?);
        }

        // 2) Common nested locations.
        for k in [
            "lm_config",
            "text_config",
            "minicpm4_config",
            "minicpm_config",
            "llm_config",
            "base_lm_config",
            "model_config",
        ] {
            if let Some(v) = self.raw.get(k) {
                if v.get("hidden_size").is_some() {
                    return Ok(parse(v).map_err(VoxCpmError::Json)?);
                }
                // Some packages embed it one level deeper.
                if let Some(obj) = v.as_object() {
                    for (_kk, vv) in obj.iter() {
                        if vv.get("hidden_size").is_some() {
                            return Ok(parse(vv).map_err(VoxCpmError::Json)?);
                        }
                    }
                }
            }
        }

        // 3) Last resort: scan top-level objects for a candidate.
        if let Some(obj) = self.raw.as_object() {
            for (_k, v) in obj.iter() {
                if v.get("hidden_size").is_some() {
                    return Ok(parse(v).map_err(VoxCpmError::Json)?);
                }
            }
        }

        let keys = self
            .raw
            .as_object()
            .map(|o| o.keys().cloned().collect::<Vec<_>>().join(", "))
            .unwrap_or_else(|| "<non-object json>".into());
        Err(VoxCpmError::InvalidArg(format!(
            "cannot locate MiniCPM4 config in config.json (expected hidden_size/num_hidden_layers/etc). top-level keys: {keys}"
        )))
    }

    /// Extract the AudioVAE config used by waveform encode/decode.
    pub fn audiovae(&self) -> Result<crate::model::audiovae::AudioVaeConfig> {
        use crate::model::audiovae::AudioVaeConfig;

        fn parse(v: &Value) -> core::result::Result<AudioVaeConfig, serde_json::Error> {
            serde_json::from_value::<AudioVaeConfig>(v.clone())
        }

        for k in [
            "audiovae_config",
            "audio_vae_config",
            "audiovae",
            "audio_vae",
            "vae_config",
            "vae",
        ] {
            if let Some(v) = self.raw.get(k) {
                if v.get("latent_dim").is_some() {
                    return Ok(parse(v).map_err(VoxCpmError::Json)?);
                }
            }
        }

        let keys = self
            .raw
            .as_object()
            .map(|o| o.keys().cloned().collect::<Vec<_>>().join(", "))
            .unwrap_or_else(|| "<non-object json>".into());
        Err(VoxCpmError::InvalidArg(format!(
            "cannot locate AudioVAE config in config.json (expected latent_dim/decoder_rates/etc). top-level keys: {keys}"
        )))
    }

    pub fn patch_size(&self) -> Option<usize> {
        self.raw.get("patch_size")?.as_u64().map(|v| v as usize)
    }

    pub fn feat_dim(&self) -> Option<usize> {
        self.raw.get("feat_dim")?.as_u64().map(|v| v as usize)
    }

    pub fn dit_mean_mode(&self) -> Option<bool> {
        self.raw.get("dit_mean_mode")?.as_bool()
    }

    pub fn residual_lm_num_layers(&self) -> usize {
        self.raw
            .get("residual_lm_num_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(6)
    }

    pub fn scalar_quantization_latent_dim(&self) -> usize {
        self.raw
            .get("scalar_quantization_latent_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(256)
    }

    pub fn scalar_quantization_scale(&self) -> i64 {
        self.raw
            .get("scalar_quantization_scale")
            .and_then(|v| v.as_i64())
            .unwrap_or(9)
    }

    pub fn max_length(&self) -> usize {
        self.raw
            .get("max_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4096)
    }

    pub fn encoder_config(&self) -> Result<VoxCpmEncoderConfig> {
        let v = self.raw.get("encoder_config").ok_or_else(|| {
            VoxCpmError::InvalidArg("missing encoder_config in config.json".into())
        })?;
        serde_json::from_value::<VoxCpmEncoderConfig>(v.clone()).map_err(VoxCpmError::Json)
    }

    pub fn dit_config(&self) -> Result<VoxCpmDitConfig> {
        let v = self
            .raw
            .get("dit_config")
            .ok_or_else(|| VoxCpmError::InvalidArg("missing dit_config in config.json".into()))?;
        serde_json::from_value::<VoxCpmDitConfig>(v.clone()).map_err(VoxCpmError::Json)
    }

    /// MiniCPM4 config for the local encoder (feat_encoder).
    pub fn locenc_minicpm4(&self) -> Result<crate::model::minicpm4::MiniCpmConfig> {
        let mut cfg = self.minicpm4()?;
        let enc = self.encoder_config()?;
        cfg.hidden_size = enc.hidden_dim;
        cfg.intermediate_size = enc.ffn_dim;
        cfg.num_attention_heads = enc.num_heads;
        cfg.num_hidden_layers = enc.num_layers;
        cfg.kv_channels = enc.kv_channels;
        cfg.vocab_size = 0;
        Ok(cfg)
    }

    /// MiniCPM4 config for the local DiT estimator (feat_decoder.estimator).
    pub fn locdit_minicpm4(&self) -> Result<crate::model::minicpm4::MiniCpmConfig> {
        let mut cfg = self.minicpm4()?;
        let dit = self.dit_config()?;
        cfg.hidden_size = dit.hidden_dim;
        cfg.intermediate_size = dit.ffn_dim;
        cfg.num_attention_heads = dit.num_heads;
        cfg.num_hidden_layers = dit.num_layers;
        cfg.kv_channels = dit.kv_channels;
        cfg.vocab_size = 0;
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxCpmEncoderConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxCpmDitConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: Option<usize>,

    pub cfm_config: crate::model::unified_cfm::CfmConfig,
}
