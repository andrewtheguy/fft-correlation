use fft_correlation::{fft_correlate_1d, Mode};
use std::fs;
use std::path::{Path, PathBuf};

const EXPECTED_SAMPLE_RATE: u32 = 8_000;

struct WavData {
    sample_rate: u32,
    samples: Vec<f32>,
}

fn testdata_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(relative)
}

fn read_pcm16_mono_wav(path: &Path) -> WavData {
    let bytes =
        fs::read(path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    assert!(
        bytes.len() >= 12,
        "{} is too short to be a WAV file",
        path.display()
    );
    assert_eq!(
        &bytes[0..4],
        b"RIFF",
        "{} is missing RIFF header",
        path.display()
    );
    assert_eq!(
        &bytes[8..12],
        b"WAVE",
        "{} is missing WAVE header",
        path.display()
    );

    let mut offset = 12;
    let mut format = None;
    let mut data = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_len =
            u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        let chunk_end = offset + chunk_len;
        assert!(
            chunk_end <= bytes.len(),
            "{} has a truncated {:?} chunk",
            path.display(),
            std::str::from_utf8(chunk_id).unwrap_or("????")
        );

        match chunk_id {
            b"fmt " => {
                assert!(
                    chunk_len >= 16,
                    "{} has an invalid fmt chunk",
                    path.display()
                );
                let audio_format =
                    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                let channels =
                    u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap());
                let sample_rate =
                    u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
                let bits_per_sample =
                    u16::from_le_bytes(bytes[offset + 14..offset + 16].try_into().unwrap());
                format = Some((audio_format, channels, sample_rate, bits_per_sample));
            }
            b"data" => {
                data = Some(&bytes[offset..chunk_end]);
            }
            _ => {}
        }

        offset = chunk_end + (chunk_len % 2);
    }

    let (audio_format, channels, sample_rate, bits_per_sample) = format.expect("missing fmt chunk");
    assert_eq!(audio_format, 1, "{} must be PCM", path.display());
    assert_eq!(channels, 1, "{} must be mono", path.display());
    assert_eq!(bits_per_sample, 16, "{} must be 16-bit PCM", path.display());

    let data = data.expect("missing data chunk");
    assert_eq!(
        data.len() % 2,
        0,
        "{} has odd-sized sample data",
        path.display()
    );

    let samples = data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32_768.0)
        .collect();

    WavData {
        sample_rate,
        samples,
    }
}

fn strongest_match_start(signal: &[f32], template: &[f32]) -> usize {
    let correlation = fft_correlate_1d(signal, template, Mode::Full).unwrap();
    let lag_offset = template.len().saturating_sub(1);
    let peak_index = correlation
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    peak_index.saturating_sub(lag_offset)
}

fn strongest_match_starts(signal: &[f32], template: &[f32], count: usize) -> Vec<usize> {
    let mut correlation = fft_correlate_1d(signal, template, Mode::Full).unwrap();
    let lag_offset = template.len().saturating_sub(1);
    let suppression_radius = template.len().max(1);
    let mut starts = Vec::with_capacity(count);

    for _ in 0..count {
        let peak_index = correlation
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        starts.push(peak_index.saturating_sub(lag_offset));

        let suppress_start = peak_index.saturating_sub(suppression_radius);
        let suppress_end = (peak_index + suppression_radius + 1).min(correlation.len());
        correlation[suppress_start..suppress_end].fill(f32::NEG_INFINITY);
    }

    starts.sort_unstable();
    starts
}

fn assert_close_in_samples(actual: usize, expected: usize, tolerance: usize, label: &str) {
    let delta = actual.abs_diff(expected);
    assert!(
        delta <= tolerance,
        "{label}: expected sample {expected}, got {actual} (delta {delta})"
    );
}

#[test]
fn detects_rthk_beep_positions_from_real_wav_assets() {
    let pattern = read_pcm16_mono_wav(&testdata_path("clips/rthk_beep.wav"));
    let signal = read_pcm16_mono_wav(&testdata_path("rthk_section_with_beep.wav"));

    assert_eq!(pattern.sample_rate, EXPECTED_SAMPLE_RATE);
    assert_eq!(signal.sample_rate, EXPECTED_SAMPLE_RATE);

    let starts = strongest_match_starts(&signal.samples, &pattern.samples, 2);
    assert_eq!(starts.len(), 2);

    assert_close_in_samples(starts[0], 11_332, 8, "first RTHK beep");
    assert_close_in_samples(starts[1], 19_353, 8, "second RTHK beep");
}

#[test]
fn detects_cbs_news_position_from_real_wav_assets() {
    let pattern = read_pcm16_mono_wav(&testdata_path("clips/cbs_news.wav"));
    let signal = read_pcm16_mono_wav(&testdata_path("cbs_news_audio_section.wav"));

    assert_eq!(pattern.sample_rate, EXPECTED_SAMPLE_RATE);
    assert_eq!(signal.sample_rate, EXPECTED_SAMPLE_RATE);

    let start = strongest_match_start(&signal.samples, &pattern.samples);
    assert_close_in_samples(start, 207_190, 16, "CBS news clip");
}
