import type { VoxcpmStage } from '../types'

const STAGES: Array<{ id: VoxcpmStage; label: string }> = [
  { id: 'download', label: 'Download' },
  { id: 'load', label: 'Load' },
]

export function StageRail(props: { stage: string | null }) {
  // The progress UI treats the pipeline as two phases.
  // Map backend stages into: Download (download/verify), Load (load), Done (ready).
  const idx = (() => {
    const s = props.stage
    if (s == null) return -1
    if (s === 'download' || s === 'verify') return 0
    if (s === 'load') return 1
    if (s === 'ready') return 2
    return -1
  })()

  return (
    <div className="stageRail" aria-label="Stages" role="list">
      {STAGES.map((s, i) => {
        const state = idx === -1 ? 'todo' : i < idx ? 'done' : i === idx ? 'active' : 'todo'
        return (
          <div
            key={s.id}
            className={`stageItem stageItem-${state}`}
            role="listitem"
            aria-current={state === 'active' ? 'step' : undefined}
          >
            <div className="stageDot" />
            <div className="stageLabel">{s.label}</div>
          </div>
        )
      })}
    </div>
  )
}
