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

pub(crate) fn encode_wav_pcm16_mono(sample_rate: u32, pcm_f32: &[f32]) -> Result<Vec<u8>, String> {
    // NOTE: hound writes WAVEFORMATEXTENSIBLE for float32, and its default channel mask for
    // 1-channel extensible WAV is FRONT_LEFT, which makes some players (e.g. macOS QuickTime)
    // route audio to only one ear. 16-bit PCM uses the legacy WAV header (no channel mask).
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    fn f32_to_i16(s: f32) -> i16 {
        let s = if s.is_finite() { s } else { 0.0 };
        // Scale to i16. Use 32768 so -1.0 can reach -32768, then clamp.
        let v = (s.clamp(-1.0, 1.0) * 32768.0).round();
        v.clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }

    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut w = hound::WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
        for s in pcm_f32.iter().copied() {
            w.write_sample::<i16>(f32_to_i16(s))
                .map_err(|e| e.to_string())?;
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

    #[test]
    fn encode_wav_pcm16_mono_is_legacy_pcm() {
        let pcm = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_wav_pcm16_mono(44_100, &pcm).unwrap();

        // Parse the fmt chunk and assert it's plain PCM (format tag 1).
        let fmt = bytes
            .windows(4)
            .position(|w| w == b"fmt ")
            .expect("fmt chunk not found");
        assert!(fmt + 24 < bytes.len());
        let format_tag = u16::from_le_bytes([bytes[fmt + 8], bytes[fmt + 9]]);
        let channels = u16::from_le_bytes([bytes[fmt + 10], bytes[fmt + 11]]);
        let sample_rate = u32::from_le_bytes([
            bytes[fmt + 12],
            bytes[fmt + 13],
            bytes[fmt + 14],
            bytes[fmt + 15],
        ]);
        let bits_per_sample = u16::from_le_bytes([bytes[fmt + 22], bytes[fmt + 23]]);
        assert_eq!(format_tag, 1);
        assert_eq!(channels, 1);
        assert_eq!(sample_rate, 44_100);
        assert_eq!(bits_per_sample, 16);

        // Round-trip decode to ensure sample count is correct.
        let mut r = hound::WavReader::new(Cursor::new(bytes)).unwrap();
        let spec = r.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 44_100);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        let samples: Vec<i16> = r.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), pcm.len());
    }
}
