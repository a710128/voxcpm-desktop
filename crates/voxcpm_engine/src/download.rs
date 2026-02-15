use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio::io::AsyncWriteExt;

use crate::ipc::{send_error, OutTx};

use voxcpm_ipc::{
    DownloadModelRequest, DownloadProgress, EngineEvent, EngineOp, EngineResponse, EngineToHost,
    JobId, LogLevel,
};

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

async fn check_remote_accessible(
    client: &reqwest::Client,
    endpoint: &str,
    token: Option<&str>,
    repo_id: &str,
    revision: &str,
    file_path: &str,
) -> Result<(), String> {
    let repo_id = repo_id.trim().trim_matches('/');
    let revision = revision.trim().trim_matches('/');
    let file_path = file_path.trim().trim_start_matches('/');
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint.trim_end_matches('/'),
        repo_id,
        revision,
        file_path
    );

    let mut head = client.head(&url).timeout(Duration::from_secs(5));
    if let Some(t) = token {
        head = head.bearer_auth(t);
    }
    let resp = head.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("head failed {} for {}", resp.status(), url));
    }
    Ok(())
}

async fn check_local_cache_and_emit(out_tx: &OutTx, job_id: JobId, root: &PathBuf) -> bool {
    let _ = out_tx
        .send(EngineToHost::Event(EngineEvent::Log {
            level: LogLevel::Warn,
            message: "download fallback: checking local cache".to_string(),
        }))
        .await;

    let req_files = [
        ("config.json", 0u32),
        ("tokenizer.json", 1u32),
        ("model.safetensors", 2u32),
    ];

    let mut ok = true;
    for (name, done) in req_files {
        let exists = tokio::fs::try_exists(&root.join(name))
            .await
            .unwrap_or(false);
        ok &= exists;
        let _ = out_tx
            .send(EngineToHost::Event(EngineEvent::DownloadProgress(
                DownloadProgress {
                    job_id,
                    file: name.to_string(),
                    done,
                    total: 4,
                    bytes_downloaded: 0,
                    bytes_total: None,
                    percent: Some(if exists { 100.0 } else { 0.0 }),
                },
            )))
            .await;
    }

    // AudioVAE: either safetensors or pth.
    let vae_st = root.join("audiovae.safetensors");
    let vae_pth = root.join("audiovae.pth");
    let st_exists = tokio::fs::try_exists(&vae_st).await.unwrap_or(false);
    let pth_exists = tokio::fs::try_exists(&vae_pth).await.unwrap_or(false);
    ok &= st_exists || pth_exists;
    let audio_name = if st_exists {
        "audiovae.safetensors"
    } else {
        "audiovae.pth"
    };
    let _ = out_tx
        .send(EngineToHost::Event(EngineEvent::DownloadProgress(
            DownloadProgress {
                job_id,
                file: audio_name.to_string(),
                done: 3,
                total: 4,
                bytes_downloaded: 0,
                bytes_total: None,
                percent: Some(if st_exists || pth_exists { 100.0 } else { 0.0 }),
            },
        )))
        .await;

    let _ = out_tx
        .send(EngineToHost::Event(EngineEvent::Log {
            level: if ok { LogLevel::Info } else { LogLevel::Error },
            message: if ok {
                "download fallback: local cache looks complete".to_string()
            } else {
                "download fallback: local cache is incomplete".to_string()
            },
        }))
        .await;

    ok
}

async fn download_one(
    out_tx: &OutTx,
    client: &reqwest::Client,
    endpoint: &str,
    token: Option<&str>,
    job_id: JobId,
    repo_id: &str,
    revision: &str,
    file_path: &str,
    local_root: &PathBuf,
    file_done: u32,
    file_total: u32,
) -> Result<(), String> {
    if file_path.contains("..") || file_path.starts_with('/') || file_path.starts_with('\\') {
        return Err(format!("invalid file_path: {file_path}"));
    }

    let repo_id = repo_id.trim().trim_matches('/');
    let revision = revision.trim().trim_matches('/');
    let file_path = file_path.trim().trim_start_matches('/');
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint.trim_end_matches('/'),
        repo_id,
        revision,
        file_path
    );

    // HEAD for etag.
    let mut head = client.head(&url);
    if let Some(t) = token {
        head = head.bearer_auth(t);
    }
    let head_resp = head.send().await.map_err(|e| e.to_string())?;
    if !head_resp.status().is_success() {
        return Err(format!("head failed {} for {}", head_resp.status(), url));
    }
    let head_bytes_total = head_resp.content_length();
    let etag = head_resp
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let local_path = local_root.join(file_path);
    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| e.to_string())?;
    }

    let etag_path = local_path.with_file_name(format!(
        "{}.etag",
        local_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("file")
    ));

    if tokio::fs::try_exists(&local_path).await.unwrap_or(false) {
        if let (Some(server), Ok(local)) =
            (etag.as_deref(), tokio::fs::read_to_string(&etag_path).await)
        {
            if local.trim() == server {
                let _ = out_tx.try_send(EngineToHost::Event(EngineEvent::DownloadProgress(
                    DownloadProgress {
                        job_id,
                        file: file_path.to_string(),
                        done: file_done,
                        total: file_total,
                        bytes_downloaded: 0,
                        bytes_total: None,
                        percent: Some(100.0),
                    },
                )));
                return Ok(());
            }
        } else if etag.is_none() {
            // Can't validate: keep existing file.
            let _ = out_tx.try_send(EngineToHost::Event(EngineEvent::DownloadProgress(
                DownloadProgress {
                    job_id,
                    file: file_path.to_string(),
                    done: file_done,
                    total: file_total,
                    bytes_downloaded: 0,
                    bytes_total: None,
                    percent: Some(100.0),
                },
            )));
            return Ok(());
        }
    }

    let mut get = client.get(&url);
    if let Some(t) = token {
        get = get.bearer_auth(t);
    }
    let resp = get.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("get failed {} for {}", resp.status(), url));
    }

    let bytes_total = resp.content_length().or(head_bytes_total);
    let _ = out_tx.try_send(EngineToHost::Event(EngineEvent::DownloadProgress(
        DownloadProgress {
            job_id,
            file: file_path.to_string(),
            done: file_done,
            total: file_total,
            bytes_downloaded: 0,
            bytes_total,
            percent: bytes_total.map(|_| 0.0),
        },
    )));

    let tmp_path = local_path.with_file_name(format!(
        "{}.tmp",
        local_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("file")
    ));
    let mut f = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| e.to_string())?;

    let mut stream = resp.bytes_stream();
    use futures_util::StreamExt;
    let mut downloaded: u64 = 0;
    let mut last_emit = Instant::now();
    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| e.to_string())?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        f.write_all(&chunk).await.map_err(|e| e.to_string())?;

        if last_emit.elapsed() >= Duration::from_millis(200) {
            last_emit = Instant::now();
            let percent = bytes_total.map(|tot| {
                if tot == 0 {
                    0.0
                } else {
                    ((downloaded as f64) * 100.0 / (tot as f64)).min(100.0) as f32
                }
            });
            let _ = out_tx.try_send(EngineToHost::Event(EngineEvent::DownloadProgress(
                DownloadProgress {
                    job_id,
                    file: file_path.to_string(),
                    done: file_done,
                    total: file_total,
                    bytes_downloaded: downloaded,
                    bytes_total,
                    percent,
                },
            )));
        }
    }
    f.flush().await.map_err(|e| e.to_string())?;
    drop(f);
    tokio::fs::rename(&tmp_path, &local_path)
        .await
        .map_err(|e| e.to_string())?;

    if let Some(etag) = etag.as_deref() {
        tokio::fs::write(&etag_path, etag)
            .await
            .map_err(|e| e.to_string())?;
    }

    let _ = out_tx.try_send(EngineToHost::Event(EngineEvent::DownloadProgress(
        DownloadProgress {
            job_id,
            file: file_path.to_string(),
            done: file_done,
            total: file_total,
            bytes_downloaded: downloaded,
            bytes_total,
            percent: bytes_total.map(|_| 100.0),
        },
    )));

    let _ = out_tx
        .send(EngineToHost::Event(EngineEvent::Log {
            level: LogLevel::Info,
            message: format!("downloaded {file_path}"),
        }))
        .await;

    Ok(())
}

pub(crate) async fn handle_download_model(
    out_tx: OutTx,
    req: DownloadModelRequest,
) -> Result<(), String> {
    let job_id = req.job_id;
    let _ = out_tx
        .send(EngineToHost::Ack {
            job_id,
            op: EngineOp::DownloadModel,
        })
        .await;

    let cache_root = PathBuf::from(&req.cache_dir);
    if let Err(e) = tokio::fs::create_dir_all(&cache_root).await {
        send_error(
            &out_tx,
            job_id,
            EngineOp::DownloadModel,
            "io",
            e.to_string(),
            true,
        )
        .await;
        return Ok(());
    }

    let endpoint = req
        .endpoint
        .as_deref()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("HF_ENDPOINT")
                .ok()
                .map(|s| s.trim().trim_end_matches('/').to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string());
    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let client = match reqwest::Client::builder().build() {
        Ok(c) => c,
        Err(e) => {
            send_error(
                &out_tx,
                job_id,
                EngineOp::DownloadModel,
                "http_client",
                e.to_string(),
                true,
            )
            .await;
            return Ok(());
        }
    };

    // Local layout: {cache_dir}/{file_path}
    let model_dir = cache_root.clone();

    // Preflight: if we can't even HEAD config.json, accept already-populated cache.
    if let Err(e) = check_remote_accessible(
        &client,
        &endpoint,
        token.as_deref(),
        &req.repo_id,
        &req.revision,
        "config.json",
    )
    .await
    {
        let ok_local = check_local_cache_and_emit(&out_tx, job_id, &model_dir).await;
        if ok_local {
            let _ = out_tx
                .send(EngineToHost::Response(EngineResponse::DownloadModel(
                    voxcpm_ipc::DownloadModelResponse {
                        job_id,
                        model_dir: model_dir.to_string_lossy().to_string(),
                    },
                )))
                .await;
            return Ok(());
        }
        send_error(
            &out_tx,
            job_id,
            EngineOp::DownloadModel,
            "head_failed",
            format!("preflight head config.json failed: {e}"),
            true,
        )
        .await;
        return Ok(());
    }

    let files = ["config.json", "tokenizer.json", "model.safetensors"];
    for (i, name) in files.iter().enumerate() {
        if let Err(e) = download_one(
            &out_tx,
            &client,
            &endpoint,
            token.as_deref(),
            job_id,
            &req.repo_id,
            &req.revision,
            name,
            &model_dir,
            i as u32,
            4,
        )
        .await
        {
            send_error(
                &out_tx,
                job_id,
                EngineOp::DownloadModel,
                "download",
                e,
                true,
            )
            .await;
            return Ok(());
        }
    }

    let audio_name = "audiovae.safetensors";
    let audio_res = download_one(
        &out_tx,
        &client,
        &endpoint,
        token.as_deref(),
        job_id,
        &req.repo_id,
        &req.revision,
        audio_name,
        &model_dir,
        3,
        4,
    )
    .await;
    if audio_res.is_err() {
        if let Err(e) = download_one(
            &out_tx,
            &client,
            &endpoint,
            token.as_deref(),
            job_id,
            &req.repo_id,
            &req.revision,
            "audiovae.pth",
            &model_dir,
            3,
            4,
        )
        .await
        {
            send_error(
                &out_tx,
                job_id,
                EngineOp::DownloadModel,
                "download",
                e,
                true,
            )
            .await;
            return Ok(());
        }
    }

    let _ = out_tx
        .send(EngineToHost::Response(EngineResponse::DownloadModel(
            voxcpm_ipc::DownloadModelResponse {
                job_id,
                model_dir: model_dir.to_string_lossy().to_string(),
            },
        )))
        .await;

    Ok(())
}
