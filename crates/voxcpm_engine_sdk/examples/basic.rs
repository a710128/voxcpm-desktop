use std::path::PathBuf;
use std::fs;
use std::time::Instant;

use voxcpm_engine_sdk::EngineSdk;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let engine = args
        .next()
        .map(PathBuf::from)
        .expect("usage: basic <engine_bin> <cache_dir> <repo_id> <text> [out_wav]");
    let cache_dir = args.next().expect("missing cache_dir");
    let repo_id = args.next().unwrap_or_else(|| "openbmb/VoxCPM1.5".to_string());
    let text = args.next().unwrap_or_else(|| "hello".to_string());
    let out_wav = args.next().unwrap_or_else(|| "out.wav".to_string());

    let sdk = EngineSdk::spawn(engine).await?;
    let mut ev = sdk.subscribe();
    tokio::spawn(async move {
        while let Ok(e) = ev.recv().await {
            eprintln!("event: {e:?}");
        }
    });

    let model_dir = sdk
        .download_model(1, repo_id, "main".to_string(), cache_dir)
        .await?;
    eprintln!("model_dir: {model_dir}");

    let model = sdk
        .load_model(2, model_dir, "cuda:5".to_string(), None, false)
        .await?;
    eprintln!("model_id: {}", model.model_id);

    let t0 = Instant::now();
    let gen = sdk
        .generate(3, model.model_id, text, None, 42, 10, 2.0, 200)
        .await?;
    let elapsed = t0.elapsed();
    eprintln!("wav bytes: {}", gen.wav_bytes.len());

    let audio_sec = if gen.sample_rate == 0 {
        0.0
    } else {
        (gen.samples as f64) / (gen.sample_rate as f64)
    };
    if audio_sec > 0.0 {
        let rtf = elapsed.as_secs_f64() / audio_sec;
        eprintln!(
            "elapsed: {:.3}s audio: {:.3}s rtf: {:.3}",
            elapsed.as_secs_f64(),
            audio_sec,
            rtf
        );
    } else {
        eprintln!("elapsed: {:.3}s audio: 0.000s rtf: n/a", elapsed.as_secs_f64());
    }

    fs::write(&out_wav, &gen.wav_bytes)?;
    eprintln!("saved wav to: {out_wav}");

    Ok(())
}
