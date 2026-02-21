import type { DownloadEventPayload } from '../types'

import { StageRail } from '../components/StageRail'
import { Modal } from '../components/Modal'

const DOWNLOAD_WEIGHT = 0.9

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
  const percent = props.download?.stage === 'downloading' ? props.download.percent : null
  const downloadIndeterminate = props.download?.stage === 'downloading' && percent == null

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
    if (percent == null) return null
    return Math.min(100, Math.max(0, percent * DOWNLOAD_WEIGHT))
  })()

  const fillWidth = (() => {
    if (phase === 'done') return '100%'
    if (phase === 'load') return `${DOWNLOAD_WEIGHT * 100}%`
    if (percent == null) return '0%'
    return `${Math.min(100, Math.max(0, percent * DOWNLOAD_WEIGHT))}%`
  })()

  const indeterminate = (() => {
    if (phase === 'load') return { leftPct: DOWNLOAD_WEIGHT * 100, widthPct: (1 - DOWNLOAD_WEIGHT) * 100, kind: 'load' as const }
    if (downloadIndeterminate) return { leftPct: 0, widthPct: DOWNLOAD_WEIGHT * 100, kind: 'download' as const }
    return null
  })()

  const backDisabled = phase === 'load' && props.error == null

  return (
    <div className="container containerNarrow">
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

      <div className="grid2">
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
                  {phase === 'download' && percent != null ? `${percent.toFixed(1)}%` : phase === 'load' ? 'Loading…' : ''}
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
                      : percent == null
                        ? 'Downloading'
                        : `Downloading ${percent.toFixed(1)}%`
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
                </div>
              ) : null}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="cardHeader">
            <div>
              <div className="h2">Log</div>
              <div className="muted">Engine output</div>
            </div>
          </div>
          <div className="cardBody">
            <pre className="pre log" tabIndex={0} role="region" aria-label="Engine log">
              {props.log}
            </pre>
          </div>
        </div>
      </div>

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
