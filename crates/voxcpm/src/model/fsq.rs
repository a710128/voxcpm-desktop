//! Scalar quantization layer (FSQ-style).
//!
//! Reference: `VoxCPM/src/voxcpm/modules/layers/scalar_quantization_layer.py`.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Debug)]
pub struct ScalarQuantizationLayer {
    in_proj: Linear,
    out_proj: Linear,
    scale: f64,
}

impl ScalarQuantizationLayer {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        latent_dim: usize,
        scale: i64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_proj = candle_nn::linear(in_dim, latent_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear(latent_dim, out_dim, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            scale: scale as f64,
        })
    }

    /// hidden: [..., in_dim] -> [..., out_dim]
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = self.in_proj.forward(hidden)?;
        let h = h.tanh()?;

        // Inference path: hard round-to-grid.
        let h = (&h * self.scale)?;
        let h = h.round()?;
        let h = (&h / self.scale)?;
        self.out_proj.forward(&h)
    }
}
