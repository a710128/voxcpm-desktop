use crate::{Result, VoxCpmError, WavInput};

/// Load prompt audio as mono f32 PCM and resample to `target_sr`.
///
/// Note: For now, `WavInput::Path` supports WAV only.
pub(crate) fn load_prompt_mono_f32(input: &WavInput<'_>, target_sr: u32) -> Result<Vec<f32>> {
    match input {
        WavInput::Samples {
            pcm_f32,
            sample_rate,
        } => {
            let mono = pcm_f32.to_vec();
            if *sample_rate == target_sr {
                Ok(mono)
            } else {
                resample_mono_rubato(&mono, *sample_rate, target_sr)
            }
        }
        WavInput::Path(p) => {
            let (interleaved, src_sr, channels) = read_wav_to_f32_interleaved(p)?;
            let mono = mixdown_to_mono(&interleaved, channels);
            if src_sr == target_sr {
                Ok(mono)
            } else {
                resample_mono_rubato(&mono, src_sr, target_sr)
            }
        }
    }
}

fn read_wav_to_f32_interleaved(path: &std::path::Path) -> Result<(Vec<f32>, u32, usize)> {
    let mut r = hound::WavReader::open(path)
        .map_err(|e| VoxCpmError::InvalidArg(format!("open wav failed: {e}")))?;
    let spec = r.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(VoxCpmError::InvalidArg("wav channels must be >= 1".into()));
    }

    let mut out = Vec::with_capacity(r.len() as usize);
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            for s in r.samples::<f32>() {
                out.push(s.map_err(|e| VoxCpmError::InvalidArg(format!("wav read failed: {e}")))?);
            }
        }
        (hound::SampleFormat::Int, 16) => {
            let denom = i16::MAX as f32;
            for s in r.samples::<i16>() {
                out.push(
                    s.map_err(|e| VoxCpmError::InvalidArg(format!("wav read failed: {e}")))? as f32
                        / denom,
                );
            }
        }
        (hound::SampleFormat::Int, 24 | 32) => {
            let bits = spec.bits_per_sample;
            let denom = (1u64 << (bits - 1)) as f32; // 24->2^23, 32->2^31
            for s in r.samples::<i32>() {
                out.push(
                    s.map_err(|e| VoxCpmError::InvalidArg(format!("wav read failed: {e}")))? as f32
                        / denom,
                );
            }
        }
        // 8-bit PCM in WAV is typically unsigned and is rare for our use case.
        (hound::SampleFormat::Int, 8) => {
            return Err(VoxCpmError::InvalidArg(
                "8-bit PCM wav is not supported".into(),
            ));
        }
        other => {
            return Err(VoxCpmError::InvalidArg(format!(
                "unsupported wav sample format: {other:?}"
            )));
        }
    }
    Ok((out, spec.sample_rate, channels))
}

fn mixdown_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        let base = i * channels;
        for ch in 0..channels {
            acc += interleaved[base + ch];
        }
        mono.push(acc / channels as f32);
    }
    mono
}

fn resample_mono_rubato(x: &[f32], src_sr: u32, dst_sr: u32) -> Result<Vec<f32>> {
    if x.is_empty() {
        return Ok(Vec::new());
    }
    if src_sr == 0 || dst_sr == 0 {
        return Err(VoxCpmError::InvalidArg("sample_rate must be > 0".into()));
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let ratio = (dst_sr as f64) / (src_sr as f64);

    // Good defaults for speech-like signals.
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // Input chunk size in samples.
    let chunk_in = 1024usize;
    let mut r = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_in, 1)
        .map_err(|e| VoxCpmError::InvalidArg(format!("resampler init failed: {e}")))?;

    let mut y = Vec::<f32>::new();
    let mut pos = 0usize;

    while pos + chunk_in <= x.len() {
        let chunk = vec![&x[pos..pos + chunk_in]];
        let out = r
            .process(&chunk, None)
            .map_err(|e| VoxCpmError::InvalidArg(format!("resampler process failed: {e}")))?;
        y.extend_from_slice(&out[0]);
        pos += chunk_in;
    }

    if pos < x.len() {
        let chunk = vec![&x[pos..]];
        let out = r.process_partial(Some(&chunk), None).map_err(|e| {
            VoxCpmError::InvalidArg(format!("resampler process_partial failed: {e}"))
        })?;
        y.extend_from_slice(&out[0]);
    }

    // Compensate for filter delay (important for alignment).
    let delay = r.output_delay();
    if y.len() > delay {
        y.drain(..delay);
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixdown_to_mono_averages_channels() {
        // 2 frames, 2 channels: [L0,R0,L1,R1]
        let interleaved = [0.0f32, 1.0, 2.0, 3.0];
        let mono = mixdown_to_mono(&interleaved, 2);
        assert_eq!(mono, vec![0.5, 2.5]);
    }

    #[test]
    fn resample_produces_finite_output() -> Result<()> {
        // 10ms of a simple ramp at 48k -> 16k.
        let src_sr = 48_000u32;
        let dst_sr = 16_000u32;
        let n = (src_sr / 100) as usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y = resample_mono_rubato(&x, src_sr, dst_sr)?;
        assert!(!y.is_empty());
        assert!(y.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn load_prompt_samples_resamples_to_target_sr() -> Result<()> {
        let pcm: Vec<f32> = (0..480).map(|i| (i as f32) / 480f32).collect();
        let input = WavInput::Samples {
            pcm_f32: &pcm,
            sample_rate: 48_000,
        };
        let y = load_prompt_mono_f32(&input, 16_000)?;
        assert!(!y.is_empty());
        // Downsampling 48k->16k should reduce sample count (ignoring filter delay).
        assert!(y.len() < pcm.len());
        assert!(y.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
