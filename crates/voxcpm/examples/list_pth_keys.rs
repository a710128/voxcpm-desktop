use std::path::PathBuf;

use candle_core::pickle::PthTensors;

fn usage() -> ! {
    eprintln!("usage: list_pth_keys <path.pth> [state_key] [limit]\n\nExamples:\n  list_pth_keys audiovae.pth\n  list_pth_keys model.pth state_dict 200");
    std::process::exit(2)
}

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args_os();
    let _exe = args.next();
    let path: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let state_key: Option<String> = args.next().and_then(|s| s.into_string().ok());
    let limit: usize = args
        .next()
        .and_then(|s| s.into_string().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    let pth = PthTensors::new(&path, state_key.as_deref())?;
    let infos = pth.tensor_infos();
    let mut keys: Vec<_> = infos.keys().cloned().collect();
    keys.sort();
    println!("tensor keys: {}", keys.len());
    for k in keys.into_iter().take(limit) {
        let ti = &infos[&k];
        println!("{} {:?} {:?}", k, ti.layout, ti.dtype);
    }
    Ok(())
}
