// Smoke-test MiniCPM4 forward_step under CUDA graph capture/replay.
//
// Run:
//   cargo run -p voxcpm --features cuda --example cuda_graph_minicpm4_forward_step_smoke

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use std::cell::RefCell;

    use voxcpm::cuda_graph::{set_mempool_release_threshold_max, CudaGraphModule};
    use voxcpm::model::minicpm4::{MiniCpmConfig, MiniCpmModel, RopeScalingConfig};

    fn tiny_cfg() -> MiniCpmConfig {
        // Keep this aligned with `minicpm4_forward_step_smoke_cpu`.
        MiniCpmConfig {
            bos_token_id: 1,
            eos_token_id: 2,
            hidden_size: 16,
            intermediate_size: 32,
            max_position_embeddings: 16,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_scaling: RopeScalingConfig {
                r#type: "longrope".to_owned(),
                // head_dim = 4 => half = 2
                long_factor: vec![1.0, 1.0],
                short_factor: vec![1.0, 1.0],
                original_max_position_embeddings: 16,
            },
            vocab_size: 0,
            use_mup: false,
            scale_emb: 1.0,
            dim_model_base: 1,
            scale_depth: 1.0,
            rope_theta: 10_000.0,
            kv_channels: None,
        }
    }

    fn mk_x(bs: usize, hidden: usize, seed: f32, dev: &Device) -> Result<Tensor> {
        // Deterministic host-side values, then H2D.
        let n = bs * hidden;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.01 + seed).sin() * 0.5)
            .collect();
        Tensor::from_vec(v, (bs, 1, hidden), dev)
    }

    pub fn main() -> Result<()> {
        let dev = Device::new_cuda_with_stream(0)?;
        set_mempool_release_threshold_max(&dev)?;
        let cfg = tiny_cfg();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut model = MiniCpmModel::new(cfg, vb)?;

        let bs = 2usize;
        let hidden = model.cfg().hidden_size;
        let max_len = model.cfg().max_position_embeddings;
        model.setup_cache(bs, max_len)?;

        // Keep the model alive for the lifetime of the graph (weights + KV cache).
        let model = RefCell::new(model);

        let x0 = mk_x(bs, hidden, 0.1, &dev)?;
        let p0 = Tensor::from_vec(vec![0u32; bs], (bs,), &dev)?;
        let example_inputs = vec![x0, p0];

        // Capture: ins[0]=x_embed [B,1,H], ins[1]=position_id [B].
        let module = CudaGraphModule::capture(&example_inputs, |ins| {
            let mut m = model.borrow_mut();
            let y = m.forward_step(&ins[0], &ins[1])?;
            Ok(vec![y])
        })?;

        for iter in 0..3usize {
            let x = mk_x(bs, hidden, 0.2 + 0.3 * (iter as f32), module.device())?;
            let p = Tensor::from_vec(vec![0u32; bs], (bs,), module.device())?;
            let y = module.run(&[x, p])?;
            let y = &y[0];
            let (yb, ys, yh) = y.dims3()?;
            if (yb, ys, yh) != (bs, 1, hidden) {
                candle_core::bail!("unexpected output shape: got ({yb},{ys},{yh})")
            }
            eprintln!("iter {iter}: ok");
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() -> candle_core::Result<()> {
    cuda_impl::main()
}

#[cfg(not(feature = "cuda"))]
fn main() {}
