use std::io::Cursor;

pub(crate) fn decode_audio_mono_f32(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    // Hinting isn't required (symphonia probes by content), but it helps select the right demuxer.
    fn guess_ext(bytes: &[u8]) -> Option<&'static str> {
        if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
            return Some("wav");
        }
        if bytes.len() >= 4 && &bytes[0..4] == b"fLaC" {
            return Some("flac");
        }
        if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
            return Some("mp3");
        }
        // MP4-family: size (4 bytes) + 'ftyp' (4 bytes)
        if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
            return Some("mp4");
        }
        None
    }

    let mut hint = Hint::new();
    if let Some(ext) = guess_ext(audio_bytes) {
        hint.with_extension(ext);
    }

    // MediaSourceStream requires a 'static source; copy bytes so decoding can own it.
    let mss = MediaSourceStream::new(
        Box::new(Cursor::new(audio_bytes.to_vec())),
        Default::default(),
    );
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;

    let track_id = format
        .default_track()
        .ok_or_else(|| "no supported audio tracks".to_string())?
        .id;
    let codec_params = format
        .tracks()
        .iter()
        .find(|t| t.id == track_id)
        .ok_or_else(|| "missing default track".to_string())?
        .codec_params
        .clone();
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| "missing sample rate".to_string())?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let mut out_mono: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(Error::DecodeError(_)) => continue,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };

        let spec = *decoded.spec();
        let ch = spec.channels.count();
        if ch == 0 {
            return Err("decoded audio has 0 channels".to_string());
        }

        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }

        // Convert any supported sample format into interleaved f32.
        let mut sbuf = SampleBuffer::<f32>::new(frames as u64, spec);
        sbuf.copy_interleaved_ref(decoded);

        let samples = sbuf.samples();
        // Downmix by average.
        for i in 0..frames {
            let base = i * ch;
            let mut acc = 0.0f32;
            for s in &samples[base..base + ch] {
                acc += *s;
            }
            out_mono.push(acc / (ch as f32));
        }
    }

    Ok((out_mono, sample_rate))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_24bit_wav_stereo_downmixes() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };

        // Write a few frames where L = -R; mono average should be ~0.
        let mut cursor = Cursor::new(Vec::<u8>::new());
        {
            let mut w = hound::WavWriter::new(&mut cursor, spec).unwrap();
            let v1: i32 = 1 << 22;
            let v2: i32 = (1 << 22) - 1234;

            // frame 1
            w.write_sample::<i32>(v1).unwrap();
            w.write_sample::<i32>(-v1).unwrap();
            // frame 2
            w.write_sample::<i32>(v2).unwrap();
            w.write_sample::<i32>(-v2).unwrap();

            w.finalize().unwrap();
        }
        let bytes = cursor.into_inner();

        let (mono, sr) = decode_audio_mono_f32(&bytes).unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(mono.len(), 2);
        for s in mono {
            assert!(s.abs() < 1e-5);
        }
    }
}
