import { useEffect, useMemo, useRef } from 'react'

import type { ProgressEventPayload } from '../types'

export function WorkspaceScreen(props: {
  deviceSpec: string
  referenceAudioName: string | null
  referenceText: string
  targetText: string
  cfgValue: number
  inferenceSteps: number
  progress: ProgressEventPayload | null
  audioUrl: string | null
  log: string
  isGenerating: boolean
  onPickReferenceAudio: (file: File) => Promise<void>
  onChangeReferenceText: (v: string) => void
  onChangeTargetText: (v: string) => void
  onChangeCfgValue: (v: number) => void
  onChangeInferenceSteps: (v: number) => void
  onGenerate: () => void
  onStop: () => void
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const generatedSec = useMemo(() => {
    const ms = props.progress?.progress?.generated_ms
    if (ms == null) return null
    return ms / 1000
  }, [props.progress])

  const requireRefText = props.referenceAudioName != null
  const refTextOk = props.referenceText.trim().length > 0
  const canGenerate = props.targetText.trim().length > 0 && (!requireRefText || refTextOk) && !props.isGenerating

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      // Cmd/Ctrl+Enter to generate.
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        if (!canGenerate) return
        e.preventDefault()
        props.onGenerate()
        return
      }

      // Esc to stop generation.
      if (e.key === 'Escape') {
        if (!props.isGenerating) return
        e.preventDefault()
        props.onStop()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [canGenerate, props.isGenerating, props.onGenerate, props.onStop])

  return (
    <div className="container">
      <div className="topbar">
        <div className="topbarTitle">Workspace</div>
        <div className="topbarRight">
          <div className="chip">Device: {props.deviceSpec}</div>
        </div>
      </div>

      <div className="grid2">
        <div className="stack">
          <div className="card">
            <div className="cardHeader">
              <div>
                <div className="h2">Reference</div>
                <div className="muted">Optional voice prompt</div>
              </div>
            </div>
            <div className="cardBody">
              <label className="label" htmlFor="referenceAudio">
                Reference audio (WAV)
              </label>
              <input
                id="referenceAudio"
                ref={fileInputRef}
                className="input"
                type="file"
                accept="audio/wav"
                onChange={async (e) => {
                  const f = e.target.files?.[0]
                  if (!f) return
                  await props.onPickReferenceAudio(f)
                }}
              />
              {props.referenceAudioName ? <div className="muted small">{props.referenceAudioName}</div> : null}

              <div className="field">
                <label className="label" htmlFor="referenceText">
                  Reference text {requireRefText ? '(required)' : ''}
                </label>
                <textarea
                  id="referenceText"
                  className="textarea"
                  rows={4}
                  value={props.referenceText}
                  onChange={(e) => props.onChangeReferenceText(e.target.value)}
                />
                {requireRefText && !refTextOk ? (
                  <div className="error">Reference text is required when reference audio is provided.</div>
                ) : null}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="cardHeader">
              <div>
                <div className="h2">Target</div>
                <div className="muted">Text to synthesize</div>
              </div>
            </div>
            <div className="cardBody">
              <div className="field">
                <label className="label" htmlFor="targetText">
                  Target text
                </label>
                <textarea
                  id="targetText"
                  className="textarea"
                  rows={8}
                  value={props.targetText}
                  onChange={(e) => props.onChangeTargetText(e.target.value)}
                />
              </div>
              <div className="field">
                <label className="label" htmlFor="cfgValue">
                  cfg_value: {props.cfgValue.toFixed(1)}
                </label>
                <input
                  id="cfgValue"
                  className="input"
                  type="range"
                  min={1}
                  max={3}
                  step={0.1}
                  value={props.cfgValue}
                  onChange={(e) => props.onChangeCfgValue(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label className="label" htmlFor="inferenceSteps">
                  inference_steps: {props.inferenceSteps}
                </label>
                <input
                  id="inferenceSteps"
                  className="input"
                  type="range"
                  min={1}
                  max={30}
                  step={1}
                  value={props.inferenceSteps}
                  onChange={(e) => props.onChangeInferenceSteps(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="cardFooter">
              <button className="btn btnPrimary" onClick={props.onGenerate} disabled={!canGenerate}>
                Generate
              </button>
              <button className="btn btnDanger" onClick={props.onStop} disabled={!props.isGenerating}>
                Stop
              </button>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="cardHeader">
            <div>
              <div className="h2">Output</div>
              <div className="muted">Audio + progress</div>
            </div>
          </div>
          <div className="cardBody">
            <div className="progressBlock">
              <div className="progressTop">
                <div className="label">Generation</div>
                <div className="muted">
                  {generatedSec != null ? `~${generatedSec.toFixed(2)}s` : props.isGenerating ? '…' : ''}
                </div>
              </div>
              <div className="timeRuler" role="progressbar" aria-label="Generated seconds">
                <div
                  className="timeFill"
                  style={{
                    width:
                      generatedSec == null
                        ? '0%'
                        : `${Math.min(100, Math.max(0, (generatedSec / 10) * 100))}%`,
                  }}
                />
              </div>
              <div className="muted small">(visual scale: 10s = 100%)</div>
            </div>

            {props.audioUrl ? (
              <audio controls src={props.audioUrl} style={{ width: '100%' }} />
            ) : (
              <div className="muted">No audio yet.</div>
            )}

            <div className="field">
              <div className="label">Log</div>
              <pre className="pre log">{props.log}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
