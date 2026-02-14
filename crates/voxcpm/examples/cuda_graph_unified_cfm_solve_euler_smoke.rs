// Smoke-test UnifiedCfm::solve_euler under CUDA graph capture/replay.
//
// Run:
//   cargo run -p voxcpm --features cuda --example cuda_graph_unified_cfm_solve_euler_smoke

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::{DType, Device, Result, Tensor};

    use voxcpm::cuda_graph::{set_mempool_release_threshold_max, CudaGraphModule};
    use voxcpm::model::unified_cfm::{build_t_span, UnifiedCfm, VelocityEstimator};

    #[derive(Clone, Debug)]
    struct DummyEstimator {
        mode: u8,
    }

    impl VelocityEstimator for DummyEstimator {
        fn forward(
            &self,
            x: &Tensor,
            _mu: &Tensor,
            _t: &Tensor,
            _cond: &Tensor,
            _dt: &Tensor,
        ) -> Result<Tensor> {
            let (b2, c, t) = x.dims3()?;
            match self.mode {
                0 => Ok(x.clone()),
                2 => Tensor::zeros((b2, c, t), x.dtype(), x.device()),
                1 => {
                    // First half ones, second half zeros.
                    let b = b2 / 2;
                    if b * 2 != b2 {
                        candle_core::bail!("DummyEstimator(mode=1) expects even batch")
                    }
                    let ones = Tensor::ones((b, c, t), x.dtype(), x.device())?;
                    let zeros = Tensor::zeros((b, c, t), x.dtype(), x.device())?;
                    Tensor::cat(&[&ones, &zeros], 0)
                }
                _ => candle_core::bail!("unknown dummy estimator mode"),
            }
        }
    }

    fn mk_x(b: usize, c: usize, t: usize, seed: f32, dev: &Device) -> Result<Tensor> {
        let n = b * c * t;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.001 + seed).cos() * 0.25)
            .collect();
        Tensor::from_vec(v, (b, c, t), dev)
    }

    pub fn main() -> Result<()> {
        let dev = Device::new_cuda_with_stream(0)?;
        set_mempool_release_threshold_max(&dev)?;

        let (b, c, t, h, tprime) = (1usize, 2usize, 8usize, 4usize, 5usize);
        let n_steps = 4usize;
        let cfg_value = 2.5f64;

        let est = DummyEstimator { mode: 1 };
        let cfm = UnifiedCfm::new(c, est, false);

        let x0 = mk_x(b, c, t, 0.1, &dev)?;
        let t_span0 = Tensor::from_vec(build_t_span(n_steps, 0.0), (n_steps + 1,), &dev)?;
        let mu0 = Tensor::zeros((b, h), DType::F32, &dev)?;
        let cond0 = Tensor::zeros((b, c, tprime), DType::F32, &dev)?;

        let example_inputs = vec![x0, t_span0, mu0, cond0];
        let module = CudaGraphModule::capture(&example_inputs, |ins| {
            cfm.solve_euler(&ins[0], &ins[1], &ins[2], &ins[3], cfg_value, false)
        })?;

        for iter in 0..3usize {
            let x = mk_x(b, c, t, 0.2 + 0.3 * (iter as f32), module.device())?;
            let t_span =
                Tensor::from_vec(build_t_span(n_steps, 0.0), (n_steps + 1,), module.device())?;
            let mu = Tensor::zeros((b, h), DType::F32, module.device())?;
            let cond = Tensor::zeros((b, c, tprime), DType::F32, module.device())?;

            let out = module.run(&[x, t_span, mu, cond])?;
            let (ob, oc, ot) = out.dims3()?;
            if (ob, oc, ot) != (b, c, t) {
                candle_core::bail!("unexpected output shape: got ({ob},{oc},{ot})")
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
