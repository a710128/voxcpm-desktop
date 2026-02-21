import type { DownloadEventPayload } from '../types'

import { StageRail } from '../components/StageRail'

export function ProgressScreen(props: {
  stage: string | null
  stageMessage: string | null
  download: DownloadEventPayload | null
  log: string
  onBack: () => void
}) {
  const percent = props.download?.stage === 'downloading' ? props.download.percent : null
  const downloadIndeterminate = props.download?.stage === 'downloading' && percent == null

  return (
    <div className="container containerNarrow">
      <div className="topBar">
        <button className="btn btnGhost" onClick={props.onBack}>
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
                <div className="label">Download</div>
                <div className="muted">{percent != null ? `${percent.toFixed(1)}%` : ''}</div>
              </div>
              <div
                className={downloadIndeterminate ? 'progressBar indeterminate' : 'progressBar'}
                role="progressbar"
                aria-label="Download progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={percent == null ? undefined : Math.min(100, Math.max(0, percent))}
                aria-valuetext={downloadIndeterminate ? 'Downloading' : undefined}
              >
                {percent == null ? null : (
                  <div className="progressFill" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
                )}
              </div>
              {props.download?.stage === 'downloading' ? (
                <div className="muted small">{props.download.file}</div>
              ) : null}
            </div>

            {props.stage === 'load' ? (
              <div className="progressBlock">
                <div className="progressTop">
                  <div className="label">Load</div>
                  <div className="muted">…</div>
                </div>
                <div
                  className="progressBar indeterminate"
                  role="progressbar"
                  aria-label="Load progress"
                  aria-valuetext="Loading"
                />
              </div>
            ) : null}
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
    </div>
  )
}
