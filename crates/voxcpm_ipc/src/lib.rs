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
    LoadModel(LoadModelRequest),
    Generate(GenerateRequest),
    StopGenerate(StopGenerateRequest),
    Exit(ExitRequest),
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
    LoadModel,
    Generate,
    StopGenerate,
    Exit,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub job_id: JobId,
    pub model_id: String,
    pub text: String,
    pub prompt_wav: Option<Vec<u8>>,
    pub seed: u64,
    pub max_steps: usize,
    pub guidance_scale: f64,
    pub emit_every_ms: u64,
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
    LoadModel(LoadModelResponse),
    Generate(GenerateResponse),
    StopGenerate(StopGenerateResponse),
    Exit(ExitResponse),
}
