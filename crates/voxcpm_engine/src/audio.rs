use std::io::Cursor;

pub(crate) fn decode_wav_mono_f32(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    let r = hound::WavReader::new(Cursor::new(wav_bytes)).map_err(|e| e.to_string())?;
    let spec = r.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;
    if ch == 0 {
        return Err("invalid wav: channels is 0".into());
    }

    let mut out: Vec<f32> = Vec::new();

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            let mut acc: Vec<f32> = Vec::with_capacity(ch);
            for s in r.into_samples::<f32>() {
                let s = s.map_err(|e| e.to_string())?;
                acc.push(s);
                if acc.len() == ch {
                    let m = acc.iter().sum::<f32>() / (ch as f32);
                    out.push(m);
                    acc.clear();
                }
            }
        }
        (hound::SampleFormat::Int, 16) => {
            let scale = i16::MAX as f32;
            let mut acc: Vec<f32> = Vec::with_capacity(ch);
            for s in r.into_samples::<i16>() {
                let s = s.map_err(|e| e.to_string())?;
                acc.push((s as f32) / scale);
                if acc.len() == ch {
                    let m = acc.iter().sum::<f32>() / (ch as f32);
                    out.push(m);
                    acc.clear();
                }
            }
        }
        (hound::SampleFormat::Int, 32) => {
            let scale = i32::MAX as f32;
            let mut acc: Vec<f32> = Vec::with_capacity(ch);
            for s in r.into_samples::<i32>() {
                let s = s.map_err(|e| e.to_string())?;
                acc.push((s as f32) / scale);
                if acc.len() == ch {
                    let m = acc.iter().sum::<f32>() / (ch as f32);
                    out.push(m);
                    acc.clear();
                }
            }
        }
        _ => {
            return Err(format!(
                "unsupported wav format: {:?} {} bits",
                spec.sample_format, spec.bits_per_sample
            ))
        }
    }

    Ok((out, sr))
}

pub(crate) fn encode_wav_f32_mono(sample_rate: u32, pcm_f32: &[f32]) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut w = hound::WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
        for s in pcm_f32.iter().copied() {
            w.write_sample(s).map_err(|e| e.to_string())?;
        }
        w.finalize().map_err(|e| e.to_string())?;
    }
    Ok(cursor.into_inner())
}
