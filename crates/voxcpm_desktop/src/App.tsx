import { useEffect, useMemo, useState } from 'react'
import { listen } from '@tauri-apps/api/event'
import { invoke } from '@tauri-apps/api/core'

type ProgressPayload = {
  event?: string
  stage?: string
  seq?: number
  progress?: {
    steps_done: number
    step_samples: number
    sample_rate: number
    generated_samples: number
    generated_ms: number
  }
}

export default function App() {
  const [repoId, setRepoId] = useState('openbmb/VoxCPM1.5')
  const [revision, setRevision] = useState('main')
  const [modelDir, setModelDir] = useState('')
  const [text, setText] = useState('hello')
  const [promptWav, setPromptWav] = useState('')
  const [deviceSpec, setDeviceSpec] = useState<'cpu' | 'cuda:0' | 'metal:0'>('cpu')
  const [progress, setProgress] = useState<ProgressPayload | null>(null)
  const [download, setDownload] = useState<any>(null)
  const [log, setLog] = useState<string>('')

  const generatedSec = useMemo(() => {
    const ms = progress?.progress?.generated_ms
    if (ms == null) return null
    return (ms / 1000).toFixed(2)
  }, [progress])

  useEffect(() => {
    const unlistenP = listen<ProgressPayload>('voxcpm:progress', (e) => {
      setProgress(e.payload)
    })
    const unlistenD = listen<any>('voxcpm:download', (e) => {
      setDownload(e.payload)
    })
    const unlistenL = listen<string>('voxcpm:log', (e) => {
      setLog((prev) => prev + e.payload)
    })
    return () => {
      unlistenP.then((f) => f()).catch(() => {})
      unlistenD.then((f) => f()).catch(() => {})
      unlistenL.then((f) => f()).catch(() => {})
    }
  }, [])

  async function downloadModel() {
    setLog('')
    setDownload(null)
    const dir = (await invoke('ensure_model', {
      repoId,
      revision,
    })) as string
    setModelDir(dir)
    setLog((prev) => prev + `Downloaded model to: ${dir}\n`)
  }

  async function runInfer() {
    setLog('')
    setProgress(null)
    const wavBytes = (await invoke('infer', {
      modelDir,
      text,
      promptWav: promptWav.trim() === '' ? null : promptWav,
      deviceSpec,
    })) as Uint8Array

    // Save as Blob URL so the user can play it.
    const blob = new Blob([wavBytes], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    setLog((prev) => prev + `\nDone. WAV bytes=${wavBytes.byteLength} url=${url}\n`)
  }

  return (
    <main>
      <h1>VoxCPM Desktop</h1>

      <label>Hugging Face repo</label>
      <input value={repoId} onChange={(e) => setRepoId(e.target.value)} />

      <label>Revision</label>
      <input value={revision} onChange={(e) => setRevision(e.target.value)} />

      <button onClick={downloadModel}>Download Model</button>

      <h2>Download</h2>
      <pre>{JSON.stringify(download, null, 2)}</pre>

      <label>Model directory</label>
      <input value={modelDir} onChange={(e) => setModelDir(e.target.value)} placeholder="/path/to/model_dir" />

      <label>Text</label>
      <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} />

      <label>Prompt wav (optional)</label>
      <input value={promptWav} onChange={(e) => setPromptWav(e.target.value)} placeholder="/path/to/prompt.wav or empty" />

      <label>Device</label>
      <select value={deviceSpec} onChange={(e) => setDeviceSpec(e.target.value as any)}>
        <option value="cpu">cpu</option>
        <option value="cuda:0">cuda:0</option>
        <option value="metal:0">metal:0</option>
      </select>

      <button onClick={runInfer} disabled={!modelDir || !text}>Generate</button>

      <h2>Progress</h2>
      <pre>{JSON.stringify(progress, null, 2)}</pre>
      {generatedSec != null ? <div>Generated seconds (approx): {generatedSec}</div> : null}

      <h2>Log</h2>
      <pre>{log}</pre>
    </main>
  )
}
