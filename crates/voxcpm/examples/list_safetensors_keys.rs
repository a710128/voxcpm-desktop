use std::path::PathBuf;

use candle_core::{safetensors, Device};

fn usage() -> ! {
    eprintln!("usage: list_safetensors_keys <path.safetensors> [limit]");
    std::process::exit(2)
}

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args_os();
    let _exe = args.next();
    let path: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let limit: usize = args
        .next()
        .and_then(|s| s.into_string().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);

    let dev = Device::Cpu;
    let ts = safetensors::load(&path, &dev)?;
    let mut keys: Vec<_> = ts.keys().cloned().collect();
    keys.sort();
    println!("keys: {}", keys.len());
    for k in keys.into_iter().take(limit) {
        println!("{k}");
    }
    Ok(())
}
