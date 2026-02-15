import type { VoxcpmStage } from '../types'

const STAGES: Array<{ id: VoxcpmStage; label: string }> = [
  { id: 'download', label: 'Download' },
  { id: 'verify', label: 'Verify' },
  { id: 'load', label: 'Load' },
  { id: 'ready', label: 'Ready' },
]

export function StageRail(props: { stage: string | null }) {
  const idx = STAGES.findIndex((s) => s.id === props.stage)

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
