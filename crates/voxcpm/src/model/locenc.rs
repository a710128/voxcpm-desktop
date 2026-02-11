//! VoxCPM local encoder (prompt feature encoder).
//!
//! Reference: `VoxCPM/src/voxcpm/modules/locenc/local_encoder.py`.

use crate::model::minicpm4::{MiniCpmConfig, MiniCpmModel};
use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Debug)]
pub struct VoxCpmLocEnc {
    special_token: Tensor, // [1, 1, 1, hidden]
    in_proj: Linear,
    encoder: MiniCpmModel,
    hidden: usize,
}

impl VoxCpmLocEnc {
    pub fn new(cfg: MiniCpmConfig, input_dim: usize, vb: VarBuilder) -> Result<Self> {
        if cfg.vocab_size != 0 {
            candle_core::bail!("locenc expects vocab_size == 0, got {}", cfg.vocab_size)
        }
        let hidden = cfg.hidden_size;
        let special_token = vb.get((1, 1, 1, hidden), "special_token")?;
        let in_proj = candle_nn::linear(input_dim, hidden, vb.pp("in_proj"))?;
        let encoder = MiniCpmModel::new(cfg, vb.pp("encoder"))?;
        Ok(Self {
            special_token,
            in_proj,
            encoder,
            hidden,
        })
    }

    /// x: [B, T, P, D] -> [B, T, hidden]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, p, d) = x.dims4()?;

        // Project input features to transformer hidden size.
        let x2 = x.reshape((b * t * p, d))?;
        let x2 = self.in_proj.forward(&x2)?;
        let x = x2.reshape((b, t, p, self.hidden))?;

        // Prepend a learnable CLS token for each (B, T).
        let special = self.special_token.broadcast_as((b, t, 1, self.hidden))?;
        let x = Tensor::cat(&[&special, &x], D::Minus2)?; // concat on P dim

        // Flatten (B, T) into batch.
        let seq = p + 1;
        let x = x.reshape((b * t, seq, self.hidden))?;
        let pos: Vec<u32> = (0..seq as u32).collect();
        let pos = Tensor::from_vec(pos, (seq,), x.device())?;

        // Non-causal transformer.
        let h = self.encoder.forward(&x, &pos, false)?; // [B*T, seq, hidden]
        let cls = h.narrow(1, 0, 1)?.squeeze(1)?; // [B*T, hidden]
        cls.reshape((b, t, self.hidden))
    }
}
