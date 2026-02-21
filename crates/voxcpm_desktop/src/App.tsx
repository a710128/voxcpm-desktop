import { useEffect, useMemo, useRef, useState } from 'react'

import { listen } from '@tauri-apps/api/event'
import { invoke } from '@tauri-apps/api/core'

import { Modal } from './components/Modal'
import {
  getLaunchCustomRepoId,
  getLaunchDeviceSpec,
  getLaunchMirror,
  getLaunchModelSource,
  setLaunchCustomRepoId,
  setLaunchDeviceSpec,
  setLaunchMirror,
  setLaunchModelSource,
  type LaunchModelSource,
} from './lib/settings'
import { LaunchScreen } from './screens/LaunchScreen'
import { ProgressScreen } from './screens/ProgressScreen'
import { WorkspaceScreen } from './screens/WorkspaceScreen'
import type {
  CapabilitiesResponse,
  DeviceSpec,
  DownloadEventPayload,
  ProgressEventPayload,
  StageEventPayload,
  VoxcpmStage,
} from './types'

type AppMode = 'boot' | 'select_device' | 'downloading_loading' | 'workspace' | 'error'

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms))
}

async function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let t: number | undefined
  try {
    return await Promise.race([
      p,
      new Promise<T>((_, reject) => {
        t = window.setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
      }),
    ])
  } finally {
    if (t != null) window.clearTimeout(t)
  }
}

function pickDefaultDevice(devices: DeviceSpec[]): DeviceSpec {
  // Prefer CUDA when available, else prefer Metal, else CPU.
  const cuda = devices.find((d) => d.toLowerCase().startsWith('cuda:'))
  if (cuda) return cuda
  const metal = devices.find((d) => d.toLowerCase().startsWith('metal:'))
  if (metal) return metal
  return 'cpu'
}

export default function App() {
  const [mode, setMode] = useState<AppMode>('boot')
  const [caps, setCaps] = useState<CapabilitiesResponse | null>(null)

  const [modelSource, setModelSource] = useState<LaunchModelSource>(() => getLaunchModelSource())
  const [customRepoId, setCustomRepoId] = useState<string>(() => getLaunchCustomRepoId())
  const [mirror, setMirror] = useState<boolean>(() => getLaunchMirror())
  const [deviceSpec, setDeviceSpec] = useState<DeviceSpec>(() => getLaunchDeviceSpec() ?? 'cpu')

  const [stage, setStage] = useState<VoxcpmStage | string | null>(null)
  const [stageMessage, setStageMessage] = useState<string | null>(null)
  const [download, setDownload] = useState<DownloadEventPayload | null>(null)
  const [progress, setProgress] = useState<ProgressEventPayload | null>(null)
  const [log, setLog] = useState<string>('')
  const pendingLogRef = useRef<string>('')
  const logFlushRafRef = useRef<number | null>(null)

  const MAX_LOG_CHARS = 200_000

  const [referenceAudioName, setReferenceAudioName] = useState<string | null>(null)
  const [referenceAudioBytes, setReferenceAudioBytes] = useState<Uint8Array | null>(null)
  const [referenceText, setReferenceText] = useState<string>('')
  const [targetText, setTargetText] = useState<string>('')
  const [cfgValue, setCfgValue] = useState<number>(2.0)
  const [inferenceSteps, setInferenceSteps] = useState<number>(10)

  const [isGenerating, setIsGenerating] = useState<boolean>(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const audioUrlRef = useRef<string | null>(null)

  const [showDownloadError, setShowDownloadError] = useState<null | { message: string }>(null)
  const [bootInfo, setBootInfo] = useState<string>('')

  useEffect(() => {
    function scheduleLogFlush() {
      if (logFlushRafRef.current != null) return
      logFlushRafRef.current = window.requestAnimationFrame(() => {
        logFlushRafRef.current = null
        const pending = pendingLogRef.current
        if (pending === '') return
        pendingLogRef.current = ''
        setLog((prev) => {
          let next = prev + pending
          if (next.length > MAX_LOG_CHARS) next = next.slice(next.length - MAX_LOG_CHARS)
          return next
        })
      })
    }

    const unlistenStage = listen<StageEventPayload>('voxcpm:stage', (e) => {
      setStage(e.payload.stage)
      setStageMessage(e.payload.message ?? null)
      if (e.payload.stage === 'ready') {
        // Auto transition to workspace.
        setMode('workspace')
      }
      if (e.payload.stage === 'error') {
        setShowDownloadError({ message: e.payload.message ?? 'unknown error' })
      }
    })

    const unlistenP = listen<ProgressEventPayload>('voxcpm:progress', (e) => {
      setProgress(e.payload)
    })
    const unlistenD = listen<DownloadEventPayload>('voxcpm:download', (e) => {
      setDownload(e.payload)
    })
    const unlistenL = listen<string>('voxcpm:log', (e) => {
      pendingLogRef.current += e.payload
      scheduleLogFlush()
    })

    return () => {
      unlistenStage.then((f) => f()).catch(() => {})
      unlistenP.then((f) => f()).catch(() => {})
      unlistenD.then((f) => f()).catch(() => {})
      unlistenL.then((f) => f()).catch(() => {})

      if (logFlushRafRef.current != null) {
        window.cancelAnimationFrame(logFlushRafRef.current)
        logFlushRafRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    // Boot: query capabilities.
    if (mode !== 'boot') return
    let cancelled = false
    ;(async () => {
      try {
        const tauriInjected = typeof (window as any).__TAURI__ !== 'undefined'
        setBootInfo(`tauriInjected=${tauriInjected}`)
        // Give the webview a brief moment to finish injection/initialization.
        await sleep(50)
        const c = await withTimeout(
          invoke<CapabilitiesResponse>('get_capabilities'),
          8000,
          'get_capabilities'
        )
        if (cancelled) return
        setCaps(c)

        const savedDevice = getLaunchDeviceSpec()
        const effectiveDevice = savedDevice && c.devices.includes(savedDevice) ? savedDevice : pickDefaultDevice(c.devices)
        setDeviceSpec(effectiveDevice)
        setMode('select_device')
      } catch (e) {
        if (cancelled) return
        setMode('error')
        setStageMessage(String(e))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [mode])

  useEffect(() => {
    // Keep localStorage in sync.
    setLaunchMirror(mirror)
  }, [mirror])

  useEffect(() => {
    setLaunchDeviceSpec(deviceSpec)
  }, [deviceSpec])

  useEffect(() => {
    setLaunchModelSource(modelSource)
  }, [modelSource])

  useEffect(() => {
    setLaunchCustomRepoId(customRepoId)
  }, [customRepoId])

  const effectiveRepoId = useMemo(() => {
    if (modelSource !== 'custom') return null
    const s = customRepoId.trim()
    return s === '' ? null : s
  }, [customRepoId, modelSource])

  async function startPrepare(opts?: { mirror?: boolean; deviceOverride?: DeviceSpec; repoId?: string | null }) {
    setLog('')
    pendingLogRef.current = ''
    if (logFlushRafRef.current != null) {
      window.cancelAnimationFrame(logFlushRafRef.current)
      logFlushRafRef.current = null
    }
    setDownload(null)
    setProgress(null)
    setStage(null)
    setStageMessage(null)
    setShowDownloadError(null)
    setMode('downloading_loading')

    const effectiveMirror = opts?.mirror ?? mirror
    const effectiveDevice = opts?.deviceOverride ?? deviceSpec
    const repoIdToUse = opts?.repoId ?? effectiveRepoId
    if (opts?.deviceOverride) setDeviceSpec(opts.deviceOverride)

    try {
      await invoke('prepare_default_model', {
        deviceSpec: effectiveDevice,
        mirror: effectiveMirror,
        repoId: repoIdToUse,
      })
      // Transition happens via voxcpm:stage ready.
    } catch (e) {
      setShowDownloadError({ message: String(e) })
    }
  }

  async function onGenerate() {
    setIsGenerating(true)
    setProgress(null)
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
    setAudioUrl(null)
    try {
      const wavBytes = await invoke<Uint8Array>('generate_v1', {
        deviceSpec,
        targetText,
        referenceAudioBytes: referenceAudioBytes ? Array.from(referenceAudioBytes) : null,
        referenceText: referenceText.trim() === '' ? null : referenceText,
        cfgValue,
        inferenceSteps,
      })
      // Tauri returns an Uint8Array-like object (typing may vary across TS libdefs).
      const blob = new Blob([wavBytes as any], { type: 'audio/wav' })
      const url = URL.createObjectURL(blob)
      audioUrlRef.current = url
      setAudioUrl(url)
    } catch (e) {
      setLog((prev) => prev + `\nGenerate failed: ${String(e)}\n`)
    } finally {
      setIsGenerating(false)
    }
  }

  async function onStop() {
    try {
      await invoke('stop_generate')
    } catch {
      // ignore
    }
    setIsGenerating(false)
    setProgress(null)
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
    setAudioUrl(null)
  }

  const content = useMemo(() => {
    if (mode === 'boot') {
      return (
        <div className="container">
          <div className="card">
            <div className="cardBody">
              <div className="appTitle">VoxCPM Desktop</div>
              <div>Booting…</div>
              <div className="muted small" style={{ marginTop: 8 }}>
                {bootInfo}
              </div>
            </div>
          </div>
        </div>
      )
    }

    if (mode === 'error') {
      return (
        <>
          <div className="container">
            <div className="card">
              <div className="cardBody">
                <div className="appTitle">VoxCPM Desktop</div>
              </div>
            </div>
          </div>

          <Modal
            title="Failed to boot"
            onClose={() => {
              setStageMessage(null)
              setMode('boot')
            }}
            footer={
              <button
                className="btn btnPrimary"
                onClick={() => {
                  setStageMessage(null)
                  setMode('boot')
                }}
              >
                Retry
              </button>
            }
          >
            <pre className="pre" style={{ margin: 0 }}>
              {stageMessage ?? 'unknown error'}
            </pre>
          </Modal>
        </>
      )
    }

    if (caps == null) {
      return null
    }

    if (mode === 'select_device') {
      return (
        <LaunchScreen
          caps={caps}
          modelSource={modelSource}
          customRepoId={customRepoId}
          mirror={mirror}
          deviceSpec={deviceSpec}
          onChangeModelSource={setModelSource}
          onChangeCustomRepoId={setCustomRepoId}
          onToggleMirror={setMirror}
          onChangeDeviceSpec={setDeviceSpec}
          onStartLoading={() => startPrepare()}
        />
      )
    }

    if (mode === 'downloading_loading') {
      return (
        <>
          <ProgressScreen
            stage={stage}
            stageMessage={stageMessage}
            download={download}
            log={log}
            onBack={() => setMode('select_device')}
          />
          {showDownloadError ? (
            <Modal
              title="Download failed"
              onClose={() => setShowDownloadError(null)}
              footer={
                <>
                  <button className="btn btnGhost" onClick={() => setMode('select_device')}>
                    Back
                  </button>
                  <button className="btn" onClick={() => startPrepare({ mirror: false })}>
                    Retry official
                  </button>
                  <button className="btn btnPrimary" onClick={() => startPrepare({ mirror: true })}>
                    Retry with mirror
                  </button>
                  {deviceSpec !== 'cpu' ? (
                    <button className="btn" onClick={() => startPrepare({ deviceOverride: 'cpu' })}>
                      Switch to CPU
                    </button>
                  ) : null}
                </>
              }
            >
              <div className="muted">{showDownloadError.message}</div>
            </Modal>
          ) : null}
        </>
      )
    }

    // workspace
    return (
      <WorkspaceScreen
        deviceSpec={deviceSpec}
        referenceAudioName={referenceAudioName}
        referenceText={referenceText}
        targetText={targetText}
        cfgValue={cfgValue}
        inferenceSteps={inferenceSteps}
        progress={progress}
        audioUrl={audioUrl}
        log={log}
        isGenerating={isGenerating}
        onPickReferenceAudio={async (file) => {
          setReferenceAudioName(file.name)
          const buf = await file.arrayBuffer()
          setReferenceAudioBytes(new Uint8Array(buf))
        }}
        onChangeReferenceText={setReferenceText}
        onChangeTargetText={setTargetText}
        onChangeCfgValue={setCfgValue}
        onChangeInferenceSteps={setInferenceSteps}
        onGenerate={onGenerate}
        onStop={onStop}
      />
    )
  }, [
    audioUrl,
    caps,
    cfgValue,
    deviceSpec,
    download,
    inferenceSteps,
    isGenerating,
    log,
    mirror,
    modelSource,
    mode,
    progress,
    customRepoId,
    referenceAudioName,
    referenceText,
    showDownloadError,
    stage,
    stageMessage,
    targetText,
  ])

  return content
}
