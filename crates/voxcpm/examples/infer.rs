use std::path::{Path, PathBuf};

use candle_core::Device;
use voxcpm::{GenerateArgs, VoxCpm, WavInput};

fn usage() -> ! {
    eprintln!(
        "usage: infer <model_dir> <text> [prompt_wav] [weights_prefix] [device]\n\nWrites ./out.wav\n\nExamples:\n  infer models/VoxCPM1.5 \"hello\"\n  infer models/VoxCPM1.5 \"hello\" none none cpu\n  infer models/VoxCPM1.5 \"hello\" none none cuda:0"
    );
    std::process::exit(2);
}

fn parse_device(s: &str) -> Result<Device, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Device::Cpu);
    }
    if s.eq_ignore_ascii_case("cpu") {
        return Ok(Device::Cpu);
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        let idx: usize = rest
            .parse()
            .map_err(|_| format!("invalid cuda device index: {s}"))?;
        #[cfg(feature = "cuda")]
        {
            return Device::new_cuda(idx).map_err(|e| format!("failed to create {s}: {e}"));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = idx;
            return Err("requested CUDA device but crate not built with feature 'cuda'".to_owned());
        }
    }
    Err(format!("unsupported device: {s} (use cpu or cuda:N)"))
}

fn main() {
    let mut args = std::env::args_os();
    let _exe = args.next();

    let model_dir: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let prompt: String = args
        .next()
        .and_then(|s| s.into_string().ok())
        .unwrap_or_else(|| usage());

    let prompt_wav: Option<PathBuf> = args.next().map(PathBuf::from).and_then(|p| {
        if p.as_os_str() == "none" {
            None
        } else {
            Some(p)
        }
    });

    let weights_prefix: Option<String> = args
        .next()
        .and_then(|s| s.into_string().ok())
        .and_then(|s| if s == "none" { None } else { Some(s) });

    let device_s: Option<String> = args.next().and_then(|s| s.into_string().ok());
    let device = match device_s.as_deref() {
        None => Device::Cpu,
        Some(s) => match parse_device(s) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{e}");
                usage();
            }
        },
    };

    let mut builder = VoxCpm::builder().show_progress(true).device(device);
    if let Some(p) = weights_prefix.as_deref() {
        builder = builder.weights_prefix(p);
    }

    let mut model = match VoxCpm::from_dir(&model_dir, builder) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("placeholder: would load model from {model_dir:?} ({err})");
            return;
        }
    };

    let ids = match model.tokenizer.encode_ids(&prompt) {
        Ok(ids) => ids,
        Err(err) => {
            eprintln!("failed to encode prompt ({err})");
            return;
        }
    };
    println!("loaded ok; prompt token count: {}", ids.len());

    let args = GenerateArgs {
        text: &prompt,
        prompt_wav: prompt_wav.as_deref().map(WavInput::Path),
        seed: 42,
        max_steps: 10,
        guidance_scale: 2.0,
    };

    let audio = match model.generate(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("generate failed: {e}");
            return;
        }
    };

    let out = Path::new("out.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(out, spec).expect("create out.wav");
    for s in audio.pcm_f32.iter().copied() {
        w.write_sample(s).expect("write sample");
    }
    w.finalize().expect("finalize wav");
    println!(
        "wrote {out:?} (sr={}, samples={})",
        audio.sample_rate,
        audio.pcm_f32.len()
    );
}
