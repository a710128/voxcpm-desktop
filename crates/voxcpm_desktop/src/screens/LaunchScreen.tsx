import { useMemo, useState } from 'react'

import { Modal } from '../components/Modal'
import type { CapabilitiesResponse, DeviceSpec } from '../types'
import type { LaunchModelSource } from '../lib/settings'

function friendlyDeviceLabel(spec: string): string {
  const s = spec.toLowerCase()
  if (s === 'cpu') return 'CPU'
  if (s.startsWith('metal:')) return spec
  if (s.startsWith('cuda:')) return spec
  return spec
}

function sortDevices(devices: DeviceSpec[]): DeviceSpec[] {
  const uniq = Array.from(new Set(devices.concat(['cpu'])))
  const cuda = uniq
    .filter((d) => d.toLowerCase().startsWith('cuda:'))
    .sort((a, b) => Number(a.split(':')[1] ?? 0) - Number(b.split(':')[1] ?? 0))
  const metal = uniq.filter((d) => d.toLowerCase().startsWith('metal:'))
  const cpu = uniq.filter((d) => d.toLowerCase() === 'cpu')
  const rest = uniq.filter((d) => !cuda.includes(d) && !metal.includes(d) && !cpu.includes(d))
  // Default priority (and display): GPU > Metal > CPU.
  return [...cuda, ...metal, ...cpu, ...rest]
}

export function LaunchScreen(props: {
  caps: CapabilitiesResponse

  // Model source
  modelSource: LaunchModelSource
  customRepoId: string
  mirror: boolean

  // Device
  deviceSpec: DeviceSpec

  // Updates
  onChangeModelSource: (v: LaunchModelSource) => void
  onChangeCustomRepoId: (v: string) => void
  onToggleMirror: (v: boolean) => void
  onChangeDeviceSpec: (v: DeviceSpec) => void

  // Action
  onStartLoading: () => void
}) {
  const [showRepoDialog, setShowRepoDialog] = useState(false)
  const [repoDraft, setRepoDraft] = useState('')

  const defaultRepoId = props.caps.defaultModel.repoId
  const isCustomSelected = props.modelSource === 'custom'
  const customRepoIdTrimmed = props.customRepoId.trim()

  const devices = useMemo(() => sortDevices(props.caps.devices), [props.caps.devices])

  const repoDraftTrimmed = repoDraft.trim()
  const repoDraftOk = repoDraftTrimmed.includes('/')

  const canLoad = props.modelSource === 'default' || (customRepoIdTrimmed !== '' && customRepoIdTrimmed.includes('/'))

  function openRepoDialog() {
    setRepoDraft(props.customRepoId)
    setShowRepoDialog(true)
  }

  return (
    <div className="container containerNarrow containerCenter">
      <div className="card">
        <div className="cardHeader">
          <div>
            <div className="h1">Load model</div>
            <div className="muted">Configure model source and device</div>
          </div>
        </div>

        <div className="cardBody">
          <div className="grid2">
            <div>
              <div className="h2">Model</div>

              <div className="field">
                <div className="label">Source</div>
                <div className="radioGroup">
                  <div className="radio modelSourceRow">
                    <input
                      id="modelSourceDefault"
                      type="radio"
                      name="modelSource"
                      checked={props.modelSource === 'default'}
                      onChange={() => props.onChangeModelSource('default')}
                    />
                    <label htmlFor="modelSourceDefault" style={{ flex: 1 }}>
                      Default ({defaultRepoId})
                    </label>
                  </div>

                  <div className="radio modelSourceRow">
                    <input
                      id="modelSourceCustom"
                      type="radio"
                      name="modelSource"
                      checked={isCustomSelected}
                      onChange={() => {
                        props.onChangeModelSource('custom')
                        if (customRepoIdTrimmed === '') openRepoDialog()
                      }}
                    />
                    <label htmlFor="modelSourceCustom" style={{ flex: 1 }}>
                      {customRepoIdTrimmed !== '' ? `Custom (${customRepoIdTrimmed})` : 'Custom…'}
                    </label>
                    <button
                      className="btn btnGhost"
                      type="button"
                      onClick={openRepoDialog}
                    >
                      {customRepoIdTrimmed !== '' ? 'Edit' : 'Set'}
                    </button>
                  </div>
                </div>
              </div>

              <div className="field">
                <div className="label">Download</div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={props.mirror}
                    onChange={(e) => props.onToggleMirror(e.target.checked)}
                  />
                  <span>Use mirror</span>
                </label>
              </div>
            </div>

            <div>
              <div className="h2">Device</div>

              <div className="field">
                <div className="label">Load to</div>
                <div className="radioGroup">
                  {devices.map((d) => (
                    <label key={d} className="radio">
                      <input
                        type="radio"
                        name="device"
                        value={d}
                        checked={props.deviceSpec === d}
                        onChange={() => props.onChangeDeviceSpec(d)}
                      />
                      <span>{friendlyDeviceLabel(d)}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="cardFooter" style={{ justifyContent: 'flex-end' }}>
          <button className="btn btnPrimary" onClick={props.onStartLoading} autoFocus disabled={!canLoad}>
            Load model
          </button>
        </div>
      </div>

      {showRepoDialog ? (
        <Modal
          title="Custom HuggingFace repo"
          onClose={() => {
            setShowRepoDialog(false)
            setRepoDraft('')
          }}
          footer={
            <>
              <button
                className="btn btnGhost"
                onClick={() => {
                  setShowRepoDialog(false)
                  setRepoDraft('')
                }}
              >
                Cancel
              </button>
              <button
                className="btn btnPrimary"
                onClick={() => {
                  if (!repoDraftOk) return
                  props.onChangeCustomRepoId(repoDraftTrimmed)
                  props.onChangeModelSource('custom')
                  setShowRepoDialog(false)
                  setRepoDraft('')
                }}
                disabled={!repoDraftOk}
              >
                Use this repo
              </button>
            </>
          }
        >
          <div className="label">Repo id</div>
          <input
            className="input"
            value={repoDraft}
            onChange={(e) => setRepoDraft(e.target.value)}
            placeholder={defaultRepoId}
            autoFocus
          />
          <div className="muted small" style={{ marginTop: 8 }}>
            Format: org or user slash repo
          </div>
        </Modal>
      ) : null}
    </div>
  )
}
