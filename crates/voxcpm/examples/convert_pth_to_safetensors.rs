use std::{collections::HashMap, path::PathBuf};

use candle_core::{pickle::PthTensors, safetensors, DType, Tensor};

fn usage() -> ! {
    eprintln!(
        "usage: convert_pth_to_safetensors <in.pth> <out.safetensors> [state_key]\n\nExample:\n  convert_pth_to_safetensors audiovae.pth audiovae.safetensors\n  convert_pth_to_safetensors model.pth model.safetensors state_dict"
    );
    std::process::exit(2)
}

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args_os();
    let _exe = args.next();
    let in_path: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let out_path: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    let state_key: Option<String> = args.next().and_then(|s| s.into_string().ok());

    let pth = PthTensors::new(&in_path, state_key.as_deref())?;
    let infos = pth.tensor_infos();

    let mut map: HashMap<String, Tensor> = HashMap::with_capacity(infos.len());
    for k in infos.keys() {
        let mut t = pth.get(k)?.expect("tensor_infos key must exist");
        // Candle cannot save some internal/quant dtypes; make it robust.
        if !matches!(
            t.dtype(),
            DType::F16
                | DType::BF16
                | DType::F32
                | DType::F64
                | DType::U8
                | DType::U32
                | DType::I64
                | DType::I32
                | DType::I16
        ) {
            t = t.to_dtype(DType::F16)?;
        }
        map.insert(k.clone(), t);
    }

    safetensors::save(&map, &out_path)?;
    println!("wrote {} tensors to {}", map.len(), out_path.display());
    Ok(())
}
