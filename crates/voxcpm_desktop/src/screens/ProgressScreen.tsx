import type { DownloadEventPayload } from '../types'

import { useEffect, useRef, useState } from 'react'

import { StageRail } from '../components/StageRail'
import { Modal } from '../components/Modal'

const DOWNLOAD_WEIGHT = 0.9

function formatBytesPerSec(bps: number): string {
  if (!Number.isFinite(bps) || bps <= 0) return ''
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s']
  let v = bps
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  const decimals = v >= 10 || i === 0 ? 0 : 1
  return `${v.toFixed(decimals)} ${units[i]}`
}

export function ProgressScreen(props: {
  stage: string | null
  stageMessage: string | null
  download: DownloadEventPayload | null
  log: string
  onBack: () => void
  error: { message: string } | null
  onDismissError: () => void
  onBackToLaunch: () => void
  onRetry: () => void
}) {
  const isDownloading = props.download?.stage === 'downloading'
  const filePercent = isDownloading ? props.download.percent : null
  const downloadDone = isDownloading ? props.download.done : 0
  const downloadTotal = isDownloading ? props.download.total : 0

  // Overall download percent is computed by splitting the bar evenly by file count.
  // overall = ((done + filePercent/100) / total) * 100
  const downloadOverall = (() => {
    if (!isDownloading || downloadTotal <= 0) return null
    const p = filePercent == null ? 0 : Math.min(100, Math.max(0, filePercent))
    const overall = ((downloadDone + p / 100) / downloadTotal) * 100
    return Math.min(100, Math.max(0, overall))
  })()

  const downloadIndeterminate = isDownloading && filePercent == null

  const [speedBps, setSpeedBps] = useState<number | null>(null)
  const speedSampleRef = useRef<{
    file: string
    bytesDownloaded: number
    tMs: number
    emaBps: number | null
  } | null>(null)

  useEffect(() => {
    if (!isDownloading) {
      speedSampleRef.current = null
      setSpeedBps(null)
      return
    }

    const file = props.download.file
    const bytesDownloaded = props.download.bytesDownloaded
    const tMs = performance.now()
    const last = speedSampleRef.current

    // Reset on file switch or non-monotonic counters.
    if (!last || last.file !== file || bytesDownloaded < last.bytesDownloaded) {
      speedSampleRef.current = { file, bytesDownloaded, tMs, emaBps: null }
      setSpeedBps(null)
      return
    }

    const dt = (tMs - last.tMs) / 1000
    const db = bytesDownloaded - last.bytesDownloaded
    if (dt <= 0 || db <= 0) {
      speedSampleRef.current = { file, bytesDownloaded, tMs, emaBps: last.emaBps }
      return
    }

    const instBps = db / dt
    const alpha = 0.25
    const emaBps = last.emaBps == null ? instBps : last.emaBps + alpha * (instBps - last.emaBps)

    speedSampleRef.current = { file, bytesDownloaded, tMs, emaBps }
    setSpeedBps(emaBps)
  }, [isDownloading, props.download?.file, props.download?.bytesDownloaded])

  const phase: 'download' | 'load' | 'done' = (() => {
    const s = props.stage
    if (s === 'load') return 'load'
    if (s === 'ready') return 'done'
    // Treat verify as part of download.
    return 'download'
  })()

  const overallNow = (() => {
    if (phase === 'done') return 100
    if (phase === 'load') return null
    if (downloadOverall == null) return null
    return Math.min(100, Math.max(0, downloadOverall * DOWNLOAD_WEIGHT))
  })()

  const fillWidth = (() => {
    if (phase === 'done') return '100%'
    if (phase === 'load') return `${DOWNLOAD_WEIGHT * 100}%`
    if (downloadOverall == null) return '0%'
    return `${Math.min(100, Math.max(0, downloadOverall * DOWNLOAD_WEIGHT))}%`
  })()

  const indeterminate = (() => {
    if (phase === 'load') return { leftPct: DOWNLOAD_WEIGHT * 100, widthPct: (1 - DOWNLOAD_WEIGHT) * 100, kind: 'load' as const }
    if (downloadIndeterminate) {
      const baseLeft = downloadTotal > 0 ? (downloadDone / downloadTotal) * DOWNLOAD_WEIGHT * 100 : 0
      const width = Math.max(0, DOWNLOAD_WEIGHT * 100 - baseLeft)
      return { leftPct: baseLeft, widthPct: width, kind: 'download' as const }
    }
    return null
  })()

  const backDisabled = phase === 'load' && props.error == null

  return (
    <div className="container containerNarrow progressScreen">
      <div className="topBar">
        <button
          className="btn btnGhost"
          onClick={props.onBack}
          disabled={backDisabled}
          title={backDisabled ? 'Model is loading…' : undefined}
        >
          Back
        </button>
      </div>

      <main className="progressStage">
        <div className="progressCardWrap">
          <div className="card">
          <div className="cardHeader">
            <div>
              <div className="h2">Progress</div>
              <div className="muted" aria-live="polite">
                {props.stageMessage ?? ' '}
              </div>
            </div>
          </div>
          <div className="cardBody">
            <StageRail stage={props.stage} />

            <div className="progressBlock">
              <div className="progressTop">
                <div className="label">Model</div>
                <div className="muted">
                  {phase === 'download' && downloadOverall != null ? `${downloadOverall.toFixed(1)}%` : phase === 'load' ? 'Loading…' : ''}
                </div>
              </div>
              <div
                className="progressBar progressBarCombined"
                role="progressbar"
                aria-label="Model preparation progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={overallNow == null ? undefined : overallNow}
                aria-valuetext={
                  phase === 'load'
                    ? 'Loading'
                    : phase === 'done'
                      ? 'Ready'
                      : downloadOverall == null
                        ? 'Downloading'
                        : `Downloading ${downloadOverall.toFixed(1)}%`
                }
              >
                <div className="progressFill" style={{ width: fillWidth }} />
                {indeterminate ? (
                  <div
                    className={indeterminate.kind === 'load' ? 'progressIndeterminate progressIndeterminateLoad' : 'progressIndeterminate'}
                    style={{ left: `${indeterminate.leftPct}%`, width: `${indeterminate.widthPct}%` }}
                  />
                ) : null}
              </div>
              {props.download?.stage === 'downloading' ? (
                <div className="muted small" title={props.download.file}>
                  {props.download.file}
                  {typeof props.download.done === 'number' && typeof props.download.total === 'number' && props.download.total > 0
                    ? ` · ${props.download.done}/${props.download.total}`
                    : ''}
                  {speedBps != null ? ` · ${formatBytesPerSec(speedBps)}` : ''}
                </div>
              ) : null}
            </div>
          </div>
        </div>
        </div>
      </main>

      {props.error ? (
        <Modal
          title="Model preparation failed"
          onClose={props.onDismissError}
          footer={
            <>
              <button className="btn btnGhost" onClick={props.onBackToLaunch}>
                Back
              </button>
              <button className="btn btnPrimary" onClick={props.onRetry}>
                Try Again
              </button>
            </>
          }
        >
          <div className="muted">{props.error.message}</div>
        </Modal>
      ) : null}
    </div>
  )
}
