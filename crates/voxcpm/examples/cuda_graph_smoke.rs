// Smoke-test CUDA graph capture/replay with Candle ops.
//
// Run:
//   cargo run -p voxcpm --features cuda --example cuda_graph_smoke

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::CpuStorage;
    use candle_core::{DType, Device, InplaceOp1, Layout, Result, Tensor};

    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::CudaStorage;

    use voxcpm::cuda_graph::{set_mempool_release_threshold_max, CudaGraphModule};

    struct CopyFromHostF32 {
        data: Vec<f32>,
    }

    impl CopyFromHostF32 {
        fn elem_count(&self) -> usize {
            self.data.len()
        }
    }

    impl InplaceOp1 for CopyFromHostF32 {
        fn name(&self) -> &'static str {
            "copy_from_host_f32"
        }

        fn cpu_fwd(&self, storage: &mut candle_core::CpuStorage, layout: &Layout) -> Result<()> {
            let n = layout.shape().elem_count();
            if n != self.elem_count() {
                candle_core::bail!(
                    "copy_from_host_f32: shape elem_count mismatch {n} != {}",
                    self.elem_count()
                )
            }
            if !layout.is_contiguous() {
                candle_core::bail!("copy_from_host_f32: expected contiguous layout")
            }
            let start = layout.start_offset();
            match storage {
                CpuStorage::F32(v) => {
                    let end = start + n;
                    if end > v.len() {
                        candle_core::bail!("copy_from_host_f32: out of bounds {end} > {}", v.len())
                    }
                    v[start..end].copy_from_slice(&self.data);
                    Ok(())
                }
                _ => candle_core::bail!("copy_from_host_f32: expected f32 storage"),
            }
        }

        fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
            let n = layout.shape().elem_count();
            if n != self.elem_count() {
                candle_core::bail!(
                    "copy_from_host_f32: shape elem_count mismatch {n} != {}",
                    self.elem_count()
                )
            }
            if storage.dtype() != DType::F32 {
                candle_core::bail!(
                    "copy_from_host_f32: expected f32 storage, got {:?}",
                    storage.dtype()
                )
            }
            if !layout.is_contiguous() {
                candle_core::bail!("copy_from_host_f32: expected contiguous layout")
            }

            let start = layout.start_offset();
            let end = start + n;

            // Copy into the existing allocation, preserving the device pointer.
            // Borrow the device first to avoid aliasing the mutable storage borrow.
            let dev = storage.device.clone();
            let slice = storage.as_cuda_slice_mut::<f32>()?;
            if end > slice.len() {
                candle_core::bail!("copy_from_host_f32: out of bounds {end} > {}", slice.len())
            }
            if start == 0 && end == slice.len() {
                dev.memcpy_htod(self.data.as_slice(), slice)?;
            } else {
                let mut dst = slice.slice_mut(start..end);
                dev.memcpy_htod(self.data.as_slice(), &mut dst)?;
            }
            Ok(())
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    fn flatten2(v: Vec<Vec<f32>>) -> Vec<f32> {
        v.into_iter().flatten().collect()
    }

    fn compute_cpu(a: &[f32], b: &[f32], shape: (usize, usize)) -> Result<Vec<f32>> {
        let dev = Device::Cpu;
        let a = Tensor::from_vec(a.to_vec(), shape, &dev)?;
        let b = Tensor::from_vec(b.to_vec(), shape, &dev)?;
        let c = (&a * &b)?.tanh()?;
        Ok(flatten2(c.to_vec2::<f32>()?))
    }

    pub fn main() -> Result<()> {
        // Small but non-trivial tensor.
        let shape = (256usize, 256usize);
        let n = shape.0 * shape.1;

        // Create a CUDA device with a non-default stream.
        // We'll query memory pool support from the stream's context (this decides has_async_alloc in cudarc).

        let mk_data = |seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = (i as f32) * 0.001 + seed;
                    (x.sin() * 0.5) + (x.cos() * 0.25)
                })
                .collect()
        };

        let cuda = Device::new_cuda_with_stream(0)?;
        set_mempool_release_threshold_max(&cuda)?;

        let example_a = Tensor::zeros(shape, DType::F32, &cuda)?;
        let example_b = Tensor::zeros(shape, DType::F32, example_a.device())?;
        let a0 = mk_data(0.1);
        let b0 = mk_data(0.2);
        example_a.inplace_op1(&CopyFromHostF32 { data: a0 })?;
        example_b.inplace_op1(&CopyFromHostF32 { data: b0 })?;
        let example_inputs = vec![example_a, example_b];

        let module = CudaGraphModule::capture(&example_inputs, |ins| {
            if ins.len() != 2 {
                candle_core::bail!("expected 2 inputs")
            }
            let out = (&ins[0] * &ins[1])?.tanh()?;
            Ok(vec![out])
        })?;

        for iter in 0..3usize {
            // Caller-provided inputs (same cuda device/stream), copied into captured slots.
            let a_it = mk_data(0.1 + iter as f32 * 0.3);
            let b_it = mk_data(0.2 + iter as f32 * 0.4);

            let a_in = Tensor::zeros(shape, DType::F32, module.device())?;
            let b_in = Tensor::zeros(shape, DType::F32, module.device())?;
            a_in.inplace_op1(&CopyFromHostF32 { data: a_it.clone() })?;
            b_in.inplace_op1(&CopyFromHostF32 { data: b_it.clone() })?;

            let out = module.run(&[a_in, b_in])?;
            let got = flatten2(out[0].to_vec2::<f32>()?);
            let expect = compute_cpu(&a_it, &b_it, shape)?;

            let diff = max_abs_diff(&got, &expect);
            if diff > 2e-3 {
                candle_core::bail!("max_abs_diff too large at iter {iter}: {diff}")
            }
            eprintln!("iter {iter}: ok (max_abs_diff={diff})");
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
