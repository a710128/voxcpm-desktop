use std::path::{Path, PathBuf};

use voxcpm::{GenerateArgs, VoxCpm, WavInput};

fn usage() -> ! {
    eprintln!(
        "usage: infer <model_dir> <text> [prompt_wav] [weights_prefix] [device] [--optimize]\n\nWrites ./out.wav\n\nExamples:\n  infer models/VoxCPM1.5 \"hello\"\n  infer models/VoxCPM1.5 \"hello\" none none cpu\n  infer models/VoxCPM1.5 \"hello\" none none cuda:0\n  infer models/VoxCPM1.5 \"hello\" none none cuda:0 --optimize"
    );
    std::process::exit(2);
}

fn main() {
    let mut raw: Vec<String> = std::env::args().collect();
    let _exe = raw.get(0).cloned();

    // Flags (kept minimal to avoid pulling in a CLI parser dependency).
    let mut optimize = false;
    raw.retain(|a| {
        if a == "--optimize" {
            optimize = true;
            false
        } else {
            true
        }
    });

    let mut args = raw.into_iter();
    let _exe = args.next();

    let model_dir: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let prompt: String = args.next().unwrap_or_else(|| usage());

    let prompt_wav: Option<PathBuf> = args.next().map(PathBuf::from).and_then(|p| {
        if p.as_os_str() == "none" {
            None
        } else {
            Some(p)
        }
    });

    let weights_prefix: Option<String> =
        args.next()
            .and_then(|s| if s == "none" { None } else { Some(s) });

    let device_spec: String = args.next().unwrap_or_else(|| "cpu".to_owned());

    // Disallow unknown extra args.
    if args.next().is_some() {
        usage();
    }

    let mut builder = VoxCpm::builder()
        .model_dir(model_dir.clone())
        .device_str(device_spec)
        .show_progress(true);
    if let Some(p) = weights_prefix.as_deref() {
        builder = builder.weights_prefix(p);
    }

    let mut model = match builder.build() {
        Ok(m) => m,
        Err(err) => {
            eprintln!("failed to load model from {model_dir:?} ({err})");
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

    if optimize {
        if let Err(e) = model.optimize() {
            eprintln!("optimize failed: {e}");
            return;
        }
        println!("cuda graph optimize: ok");
    }

    let args = GenerateArgs {
        text: &prompt,
        prompt_wav: prompt_wav.as_deref().map(WavInput::Path),
        seed: 42,
        max_steps: 10,
        guidance_scale: 2.0,
    };

    let gen_start = std::time::Instant::now();
    let audio = match model.generate(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("generate failed: {e}");
            return;
        }
    };
    let gen_elapsed = gen_start.elapsed();

    let audio_dur_s = (audio.pcm_f32.len() as f64) / (audio.sample_rate as f64);
    let gen_s = gen_elapsed.as_secs_f64();
    let rtf = if audio_dur_s > 0.0 {
        gen_s / audio_dur_s
    } else {
        0.0
    };
    println!(
        "generate_time_s={gen_s:.3} audio_dur_s={audio_dur_s:.3} rtf={rtf:.3} (sr={} samples={})",
        audio.sample_rate,
        audio.pcm_f32.len()
    );

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
