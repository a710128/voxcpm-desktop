mod app;
mod audio;
mod download;
mod infer_actor;
mod inference;
mod ipc;
mod state;
mod util;

#[tokio::main]
async fn main() {
    if let Err(e) = app::run().await {
        // Best-effort: stderr only.
        eprintln!("engine fatal: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::util::prod_usize;

    #[test]
    fn progress_math_uses_decoder_rates_product() {
        // hop = product(decoder_rates)
        let decoder_rates = vec![2usize, 2, 2, 2];
        let hop = prod_usize(&decoder_rates);
        assert_eq!(hop, 16);

        // step_samples = patch_size * hop
        let patch_size = 8u64;
        let step_samples = patch_size * hop as u64;
        assert_eq!(step_samples, 128);

        // generated_ms = generated_samples * 1000 / sample_rate
        let sample_rate = 24000u64;
        let steps_done = 10u64;
        let generated_samples = steps_done * step_samples;
        let generated_ms = (generated_samples as u128) * 1000u128 / (sample_rate as u128);
        assert_eq!(generated_ms as u64, 53);
    }
}
