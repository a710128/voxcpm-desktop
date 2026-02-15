import { useMemo } from 'react'

import type { CapabilitiesResponse, DeviceSpec } from '../types'

export function LaunchScreen(props: {
  caps: CapabilitiesResponse
  deviceSpec: DeviceSpec
  mirrorDefault: boolean
  onChangeDeviceSpec: (v: DeviceSpec) => void
  onToggleMirrorDefault: (v: boolean) => void
  onStartLoading: () => void
}) {
  const hasCuda = useMemo(() => props.caps.devices.some((d) => d.toLowerCase().startsWith('cuda:')), [props.caps.devices])

  const visibleDeviceChoices = useMemo(() => {
    if (hasCuda) {
      const cpuFirst = ['cpu', ...props.caps.devices.filter((d) => d !== 'cpu')]
      return Array.from(new Set(cpuFirst))
    }
    return []
  }, [hasCuda, props.caps.devices])

  return (
    <div className="container">
      <div className="topbar">
        <div className="topbarTitle">VoxCPM Desktop</div>
        <div className="topbarRight">
          <label className="toggle">
            <input
              type="checkbox"
              checked={props.mirrorDefault}
              onChange={(e) => props.onToggleMirrorDefault(e.target.checked)}
            />
            <span>Default use mirror</span>
          </label>
        </div>
      </div>

      <div className="grid2">
        <div className="card">
          <div className="cardHeader">
            <div>
              <div className="h1">Load default model</div>
              <div className="muted">
                {props.caps.defaultModel.repoId}@{props.caps.defaultModel.revision}
              </div>
            </div>
          </div>
          <div className="cardBody">
            {hasCuda ? (
              <div className="field">
                <div className="label">Device</div>
                <div className="radioGroup">
                  {visibleDeviceChoices.map((d) => (
                    <label key={d} className="radio">
                      <input
                        type="radio"
                        name="device"
                        value={d}
                        checked={props.deviceSpec === d}
                        onChange={() => props.onChangeDeviceSpec(d)}
                      />
                      <span>{d}</span>
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          <div className="cardFooter">
            <button className="btn btnPrimary" onClick={props.onStartLoading}>
              {hasCuda ? 'Start loading' : 'Load model'}
            </button>
          </div>
        </div>

        <div className="card">
          <div className="cardHeader">
            <div>
              <div className="h2">Devices</div>
              <div className="muted">Reported by sidecar</div>
            </div>
          </div>
          <div className="cardBody">
            <pre className="pre">{JSON.stringify(props.caps.devices, null, 2)}</pre>
          </div>
        </div>
      </div>
    </div>
  )
}
