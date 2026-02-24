use bytes::{Buf, BufMut, BytesMut};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub type JobId = u64;

pub const DEFAULT_MAX_FRAME_LEN: usize = 512 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error("frame too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("bincode encode failed: {0}")]
    Encode(String),
    #[error("bincode decode failed: {0}")]
    Decode(String),
}

pub fn encode_frame<T: Serialize>(msg: &T) -> Result<Vec<u8>, IpcError> {
    let payload = bincode::serialize(msg).map_err(|e| IpcError::Encode(e.to_string()))?;
    if payload.len() > (u32::MAX as usize) {
        return Err(IpcError::FrameTooLarge(payload.len()));
    }
    let mut out = Vec::with_capacity(4 + payload.len());
    out.put_u32_le(payload.len() as u32);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Try decode one frame from `buf`.
///
/// Returns:
/// - Ok(None) if not enough bytes yet
/// - Ok(Some(msg)) if a full frame was decoded
pub fn try_decode_frame<T: DeserializeOwned>(
    buf: &mut BytesMut,
    max_frame_len: usize,
) -> Result<Option<T>, IpcError> {
    if buf.len() < 4 {
        return Ok(None);
    }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if len > max_frame_len {
        return Err(IpcError::FrameTooLarge(len));
    }
    if buf.len() < 4 + len {
        return Ok(None);
    }

    buf.advance(4);
    let payload = buf.split_to(len);
    let msg = bincode::deserialize::<T>(&payload).map_err(|e| IpcError::Decode(e.to_string()))?;
    Ok(Some(msg))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostToEngine {
    DownloadModel(DownloadModelRequest),
    CancelDownload(CancelDownloadRequest),
    LoadModel(LoadModelRequest),
    Generate(GenerateRequest),
    StopGenerate(StopGenerateRequest),
    Exit(ExitRequest),
    ListDevices(ListDevicesRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineToHost {
    Ack { job_id: JobId, op: EngineOp },
    Event(EngineEvent),
    Response(EngineResponse),
    Error(EngineError),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EngineOp {
    DownloadModel,
    CancelDownload,
    LoadModel,
    Generate,
    StopGenerate,
    Exit,
    ListDevices,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineError {
    pub job_id: JobId,
    pub op: EngineOp,
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineEvent {
    Log { level: LogLevel, message: String },
    DownloadProgress(DownloadProgress),
    GenerateProgress(GenerateProgress),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelRequest {
    pub job_id: JobId,
    pub repo_id: String,
    pub revision: String,
    pub cache_dir: String,
    /// Optional HF endpoint override, e.g. https://hf-mirror.com
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub job_id: JobId,
    pub file: String,
    pub done: u32,
    pub total: u32,

    /// Bytes downloaded for the current file.
    pub bytes_downloaded: u64,

    /// Total bytes for the current file (from Content-Length), if known.
    pub bytes_total: Option<u64>,

    /// 0..=100, if total is known.
    pub percent: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelResponse {
    pub job_id: JobId,
    pub model_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelDownloadRequest {
    /// Job id for this cancel operation.
    pub job_id: JobId,
    /// The download job to cancel.
    pub target_job_id: JobId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelDownloadResponse {
    pub job_id: JobId,
    pub cancelled_job_id: Option<JobId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub job_id: JobId,
    pub model_dir: String,
    pub device_spec: String,
    pub weights_prefix: Option<String>,
    pub optimize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelResponse {
    pub job_id: JobId,
    pub model_id: String,
    pub step_samples: u64,
    pub sample_rate: u32,
}

/// Generate request.
///
/// IPC wire format (bincode) is intentionally kept stable and only contains
/// the fields needed by the engine today.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub job_id: JobId,
    pub model_id: String,
    pub text: String,

    /// Optional prompt audio bytes (previously called `prompt_wav`).
    pub prompt_audio_bytes: Option<Vec<u8>>,

    /// Reference text for prompt audio (if provided).
    pub prompt_text: Option<String>,
    pub seed: u64,
    pub max_steps: usize,
    pub guidance_scale: f64,
    pub emit_every_ms: u64,
}

// Stable IPC (bincode) representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerateRequestWire {
    job_id: JobId,
    model_id: String,
    text: String,
    prompt_audio_bytes: Option<Vec<u8>>,
    prompt_text: Option<String>,
    seed: u64,
    max_steps: usize,
    guidance_scale: f64,
    emit_every_ms: u64,
}

// Human-readable representation (JSON/etc). Supports legacy `prompt_wav`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerateRequestHuman {
    job_id: JobId,
    model_id: String,
    text: String,

    #[serde(default, alias = "prompt_wav")]
    prompt_audio_bytes: Option<Vec<u8>>,

    #[serde(default)]
    prompt_text: Option<String>,
    seed: u64,
    max_steps: usize,
    guidance_scale: f64,
    emit_every_ms: u64,
}

impl serde::Serialize for GenerateRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            let h = GenerateRequestHuman {
                job_id: self.job_id,
                model_id: self.model_id.clone(),
                text: self.text.clone(),
                prompt_audio_bytes: self.prompt_audio_bytes.clone(),
                prompt_text: self.prompt_text.clone(),
                seed: self.seed,
                max_steps: self.max_steps,
                guidance_scale: self.guidance_scale,
                emit_every_ms: self.emit_every_ms,
            };
            h.serialize(serializer)
        } else {
            let w = GenerateRequestWire {
                job_id: self.job_id,
                model_id: self.model_id.clone(),
                text: self.text.clone(),
                prompt_audio_bytes: self.prompt_audio_bytes.clone(),
                prompt_text: self.prompt_text.clone(),
                seed: self.seed,
                max_steps: self.max_steps,
                guidance_scale: self.guidance_scale,
                emit_every_ms: self.emit_every_ms,
            };
            w.serialize(serializer)
        }
    }
}

impl<'de> serde::Deserialize<'de> for GenerateRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let h = GenerateRequestHuman::deserialize(deserializer)?;
            Ok(Self {
                job_id: h.job_id,
                model_id: h.model_id,
                text: h.text,
                prompt_audio_bytes: h.prompt_audio_bytes,
                prompt_text: h.prompt_text,
                seed: h.seed,
                max_steps: h.max_steps,
                guidance_scale: h.guidance_scale,
                emit_every_ms: h.emit_every_ms,
            })
        } else {
            let w = GenerateRequestWire::deserialize(deserializer)?;
            Ok(Self {
                job_id: w.job_id,
                model_id: w.model_id,
                text: w.text,
                prompt_audio_bytes: w.prompt_audio_bytes,
                prompt_text: w.prompt_text,
                seed: w.seed,
                max_steps: w.max_steps,
                guidance_scale: w.guidance_scale,
                emit_every_ms: w.emit_every_ms,
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDevicesRequest {
    pub job_id: JobId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDevicesResponse {
    pub job_id: JobId,
    pub devices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateProgress {
    pub job_id: JobId,
    pub steps_done: usize,
    pub step_samples: u64,
    pub sample_rate: u32,
    pub generated_samples: u64,
    pub generated_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub job_id: JobId,
    pub wav_bytes: Vec<u8>,
    pub sample_rate: u32,
    pub samples: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopGenerateRequest {
    pub job_id: JobId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopGenerateResponse {
    pub job_id: JobId,
    pub cancelled_job_id: Option<JobId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitRequest {
    pub job_id: JobId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitResponse {
    pub job_id: JobId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineResponse {
    DownloadModel(DownloadModelResponse),
    CancelDownload(CancelDownloadResponse),
    LoadModel(LoadModelResponse),
    Generate(GenerateResponse),
    StopGenerate(StopGenerateResponse),
    Exit(ExitResponse),
    ListDevices(ListDevicesResponse),
}
