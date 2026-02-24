import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { open, save } from '@tauri-apps/plugin-dialog'
import { readFile, writeFile } from '@tauri-apps/plugin-fs'

import type { ProgressEventPayload } from '../types'

// Waveform rendering scale.
const WAVE_BAR_W = 2
const WAVE_GAP = 2
const WAVE_STEP = WAVE_BAR_W + WAVE_GAP
// Roughly 80px/s at 2px bars + 2px gaps.
const WAVE_BARS_PER_SEC = 20
const WAVE_MIN_BARS = 24
const WAVE_MAX_BARS = 20000

const SUPPORTED_REF_AUDIO_EXTS = ['wav', 'flac', 'mp3', 'm4a', 'aac', 'mp4'] as const
const SUPPORTED_REF_AUDIO_EXT_LIST = SUPPORTED_REF_AUDIO_EXTS.map((x) => `.${x}`).join(', ')
const REFERENCE_AUDIO_ACCEPT = SUPPORTED_REF_AUDIO_EXTS.map((x) => `.${x}`).join(',')
const SUPPORTED_REF_AUDIO_EXTS_DIALOG: string[] = [...SUPPORTED_REF_AUDIO_EXTS]

function isSupportedRefAudioName(name: string): boolean {
  const m = /\.([a-z0-9]+)$/i.exec(name.trim())
  const ext = (m?.[1] ?? '').toLowerCase()
  return (SUPPORTED_REF_AUDIO_EXTS as readonly string[]).includes(ext)
}

const SUPPORTED_REF_AUDIO_MIMES = new Set([
  // wav
  'audio/wav',
  'audio/x-wav',
  // flac
  'audio/flac',
  'audio/x-flac',
  // mp3
  'audio/mpeg',
  // m4a/mp4 containers
  'audio/mp4',
  'audio/x-m4a',
  // aac
  'audio/aac',
  'audio/x-aac',
  // mp4 is often reported as video/* even when it's audio-only.
  'video/mp4',
])

function drawWave(args: {
  canvas: HTMLCanvasElement | null
  peaks: Float32Array | null
  frac: number
  scrollLeft?: number
}) {
  const canvas = args.canvas
  if (!canvas) return
  const peaks = args.peaks

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const dpr = window.devicePixelRatio || 1
  const w = Math.max(1, canvas.clientWidth)
  const h = Math.max(1, canvas.clientHeight)
  canvas.width = Math.floor(w * dpr)
  canvas.height = Math.floor(h * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)

  if (!peaks || peaks.length === 0) return

  const frac = Math.max(0, Math.min(1, args.frac))

  const bars = peaks.length
  const barW = WAVE_BAR_W
  const step = WAVE_STEP
  const totalW = bars * barW + (bars - 1) * WAVE_GAP
  // Center short waveforms; long waveforms scroll from 0.
  const left = totalW < w ? Math.max(0, Math.floor((w - totalW) / 2)) : 0
  const maxScroll = Math.max(0, left + totalW - w)
  const scrollLeft = Math.max(0, Math.min(maxScroll, args.scrollLeft ?? 0))

  const mid = h / 2
  const playheadAbs = left + frac * totalW
  const playheadX = playheadAbs - scrollLeft

  const ampMax = Math.max(1, h - 10)
  const minBarH = 6
  function barHeight(v: number) {
    // Boost perceived height a bit.
    const x = Math.max(0, Math.min(1, v))
    const boosted = Math.pow(x, 0.65)
    return Math.max(minBarH, boosted * ampMax)
  }

  ctx.lineCap = 'round'
  ctx.lineWidth = barW

  const viewStartAbs = scrollLeft
  const viewEndAbs = scrollLeft + w
  const startIdx = Math.max(0, Math.floor((viewStartAbs - left) / step) - 2)
  const endIdx = Math.min(bars, Math.ceil((viewEndAbs - left) / step) + 2)
  const playedIdx = Math.max(-1, Math.min(bars - 1, Math.floor((playheadAbs - left) / step)))

  // Unplayed bars.
  ctx.strokeStyle = 'rgba(17, 24, 39, 0.38)'
  ctx.beginPath()
  for (let i = Math.max(startIdx, playedIdx + 1); i < endIdx; i++) {
    const x = left + i * step + barW / 2 - scrollLeft
    const barH = barHeight(peaks[i])
    ctx.moveTo(x, mid - barH / 2)
    ctx.lineTo(x, mid + barH / 2)
  }
  ctx.stroke()

  // Played bars.
  ctx.strokeStyle = 'rgba(59, 91, 253, 0.95)'
  ctx.beginPath()
  for (let i = startIdx; i < endIdx && i <= playedIdx; i++) {
    const x = left + i * step + barW / 2 - scrollLeft
    const barH = barHeight(peaks[i])
    ctx.moveTo(x, mid - barH / 2)
    ctx.lineTo(x, mid + barH / 2)
  }
  ctx.stroke()

  // Playhead.
  const px = Math.max(0, Math.min(w, playheadX))
  ctx.strokeStyle = 'rgba(185, 160, 255, 0.72)'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(px, 6)
  ctx.lineTo(px, h - 6)
  ctx.stroke()
}

function formatTime(sec: number | null) {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return '--:--'
  const s = Math.floor(sec)
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${m}:${String(r).padStart(2, '0')}`
}

function WaveformPlayer(p: {
  label: string
  src: string | null
  peaks: Float32Array | null
  canvasRef: { current: HTMLCanvasElement | null }
  isDecoding: boolean
  waveRedrawTick: number
  overlayLabel?: string | null
  onSave?: () => void
  isSaving?: boolean
  onClear?: () => void
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const scrollRafRef = useRef<number | null>(null)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [duration, setDuration] = useState<number | null>(null)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const [volume, setVolume] = useState<number>(1)
  const [isMuted, setIsMuted] = useState<boolean>(false)
  const [showVolume, setShowVolume] = useState<boolean>(false)

  const SPEEDS = [0.75, 1, 1.25, 1.5, 2] as const
  const [speedIdx, setSpeedIdx] = useState<number>(1)
  const speed = SPEEDS[speedIdx]

  const frac = useMemo(() => {
    if (duration == null || !Number.isFinite(duration) || duration <= 0) return 0
    return Math.max(0, Math.min(1, currentTime / duration))
  }, [currentTime, duration])

  // Keep the audio element volume in sync.
  useEffect(() => {
    const a = audioRef.current
    if (!a) return
    a.volume = Math.max(0, Math.min(1, volume))
  }, [volume])

  useEffect(() => {
    const a = audioRef.current
    if (!a) return
    a.muted = isMuted
  }, [isMuted])

  useEffect(() => {
    const a = audioRef.current
    if (!a) return
    a.playbackRate = speed
  }, [speed])

  useEffect(() => {
    if (!showVolume) return
    function onDown(e: PointerEvent) {
      const target = e.target as HTMLElement | null
      if (!target) return
      if (target.closest('.wfVolumeWrap')) return
      setShowVolume(false)
    }
    window.addEventListener('pointerdown', onDown, true)
    return () => window.removeEventListener('pointerdown', onDown, true)
  }, [showVolume])

  useEffect(() => {
    const aEl = audioRef.current
    if (!aEl) return

    const a: HTMLAudioElement = aEl

    function stopRaf() {
      if (rafRef.current != null) {
        window.cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }

    function fracFromEl() {
      const d = a!.duration
      const t = a!.currentTime
      if (!Number.isFinite(d) || d <= 0) return 0
      return Math.max(0, Math.min(1, t / d))
    }

    function redrawFromEl() {
      drawWave({
        canvas: p.canvasRef.current,
        peaks: p.peaks,
        frac: fracFromEl(),
        scrollLeft: scrollRef.current?.scrollLeft ?? 0,
      })
    }

    function ensurePlayheadVisible(nextFrac: number) {
      const sc = scrollRef.current
      const canvas = p.canvasRef.current
      if (!sc || !canvas || !p.peaks || p.peaks.length === 0) return
      const w = Math.max(1, canvas.clientWidth)
      const bars = p.peaks.length
      const totalW = bars * WAVE_BAR_W + (bars - 1) * WAVE_GAP
      const left = totalW < w ? Math.max(0, Math.floor((w - totalW) / 2)) : 0
      const maxScroll = Math.max(0, left + totalW - w)
      if (maxScroll <= 0) return

      const playheadAbs = left + Math.max(0, Math.min(1, nextFrac)) * totalW
      const cur = sc.scrollLeft
      const safeLeft = w * 0.25
      const safeRight = w * 0.35
      const visStart = cur
      const visEnd = cur + w
      let next = cur
      if (playheadAbs < visStart + safeLeft) {
        next = playheadAbs - safeLeft
      } else if (playheadAbs > visEnd - safeRight) {
        next = playheadAbs - (w - safeRight)
      } else {
        return
      }
      sc.scrollLeft = Math.max(0, Math.min(maxScroll, next))
    }

    function updateOnce() {
      const d = a!.duration
      setDuration(Number.isFinite(d) && d > 0 ? d : null)
      setCurrentTime(a!.currentTime || 0)
      setIsPlaying(!a!.paused && !a!.ended)
      redrawFromEl()
    }

    function tick() {
      // Redraw smoothly during playback without forcing React re-renders.
      const f = fracFromEl()
      ensurePlayheadVisible(f)
      drawWave({
        canvas: p.canvasRef.current,
        peaks: p.peaks,
        frac: f,
        scrollLeft: scrollRef.current?.scrollLeft ?? 0,
      })
      if (!a!.paused && !a!.ended) {
        rafRef.current = window.requestAnimationFrame(tick)
      } else {
        stopRaf()
      }
    }

    function onPlay() {
      stopRaf()
      rafRef.current = window.requestAnimationFrame(tick)
      setIsPlaying(true)
    }

    function onPause() {
      stopRaf()
      updateOnce()
    }

    function onEnded() {
      stopRaf()
      updateOnce()
    }

    a!.addEventListener('loadedmetadata', updateOnce)
    a!.addEventListener('timeupdate', updateOnce)
    a!.addEventListener('seeked', updateOnce)
    a!.addEventListener('play', onPlay)
    a!.addEventListener('pause', onPause)
    a!.addEventListener('ended', onEnded)
    updateOnce()

    return () => {
      stopRaf()
      a!.removeEventListener('loadedmetadata', updateOnce)
      a!.removeEventListener('timeupdate', updateOnce)
      a!.removeEventListener('seeked', updateOnce)
      a!.removeEventListener('play', onPlay)
      a!.removeEventListener('pause', onPause)
      a!.removeEventListener('ended', onEnded)
    }
  }, [p.canvasRef, p.peaks, p.src])

  useEffect(() => {
    // Redraw on resize as well.
    drawWave({ canvas: p.canvasRef.current, peaks: p.peaks, frac, scrollLeft: scrollRef.current?.scrollLeft ?? 0 })
  }, [p.waveRedrawTick, p.canvasRef, p.peaks, frac])

  useEffect(() => {
    // When loading a new waveform, start at the beginning.
    if (scrollRef.current) scrollRef.current.scrollLeft = 0
    drawWave({ canvas: p.canvasRef.current, peaks: p.peaks, frac, scrollLeft: 0 })
  }, [p.peaks])

  function togglePlay() {
    const a = audioRef.current
    if (!a || !p.src) return
    if (a.paused || a.ended) {
      void a.play().catch(() => {})
    } else {
      a.pause()
    }
  }

  function skipBySeconds(delta: number) {
    const a = audioRef.current
    if (!a) return
    const d = a.duration
    const maxT = Number.isFinite(d) && d > 0 ? d : Number.POSITIVE_INFINITY
    a.currentTime = Math.max(0, Math.min(maxT, (a.currentTime || 0) + delta))
    setCurrentTime(a.currentTime || 0)
    const f = Number.isFinite(d) && d > 0 ? a.currentTime / d : 0
    if (Number.isFinite(d) && d > 0) {
      // Keep the playhead in view after jumps.
      const sc = scrollRef.current
      const canvas = p.canvasRef.current
      if (sc && canvas && p.peaks && p.peaks.length > 0) {
        const w = Math.max(1, canvas.clientWidth)
        const bars = p.peaks.length
        const totalW = bars * WAVE_BAR_W + (bars - 1) * WAVE_GAP
        const left = totalW < w ? Math.max(0, Math.floor((w - totalW) / 2)) : 0
        const maxScroll = Math.max(0, left + totalW - w)
        const playheadAbs = left + Math.max(0, Math.min(1, f)) * totalW
        const cur = sc.scrollLeft
        const safeLeft = w * 0.25
        const safeRight = w * 0.35
        const visStart = cur
        const visEnd = cur + w
        let next = cur
        if (playheadAbs < visStart + safeLeft) next = playheadAbs - safeLeft
        else if (playheadAbs > visEnd - safeRight) next = playheadAbs - (w - safeRight)
        sc.scrollLeft = Math.max(0, Math.min(maxScroll, next))
      }
    }
    drawWave({ canvas: p.canvasRef.current, peaks: p.peaks, frac: f, scrollLeft: scrollRef.current?.scrollLeft ?? 0 })
  }

  function toggleMute() {
    setIsMuted((v) => !v)
    setShowVolume(false)
  }

  function cycleSpeed() {
    setSpeedIdx((i) => (i + 1) % SPEEDS.length)
  }

  function seekToFrac(nextFrac: number) {
    const a = audioRef.current
    if (!a) return
    const d = a.duration
    if (!Number.isFinite(d) || d <= 0) return
    const f = Math.max(0, Math.min(1, nextFrac))
    a.currentTime = f * d
    setCurrentTime(a.currentTime)
    // Ensure the playhead stays visible after seeks.
    const sc = scrollRef.current
    const canvas = p.canvasRef.current
    if (sc && canvas && p.peaks && p.peaks.length > 0) {
      const w = Math.max(1, canvas.clientWidth)
      const bars = p.peaks.length
      const totalW = bars * WAVE_BAR_W + (bars - 1) * WAVE_GAP
      const left = totalW < w ? Math.max(0, Math.floor((w - totalW) / 2)) : 0
      const maxScroll = Math.max(0, left + totalW - w)
      const playheadAbs = left + f * totalW
      const cur = sc.scrollLeft
      const safeLeft = w * 0.25
      const safeRight = w * 0.35
      const visStart = cur
      const visEnd = cur + w
      let next = cur
      if (playheadAbs < visStart + safeLeft) next = playheadAbs - safeLeft
      else if (playheadAbs > visEnd - safeRight) next = playheadAbs - (w - safeRight)
      sc.scrollLeft = Math.max(0, Math.min(maxScroll, next))
    }
    drawWave({ canvas: p.canvasRef.current, peaks: p.peaks, frac: f, scrollLeft: scrollRef.current?.scrollLeft ?? 0 })
  }

  function fracFromClientX(clientX: number) {
    const canvas = p.canvasRef.current
    if (!canvas || !p.peaks || p.peaks.length === 0) return null
    const rect = canvas.getBoundingClientRect()
    const x = clientX - rect.left
    const w = Math.max(1, rect.width)
    const bars = p.peaks.length
    const totalW = bars * WAVE_BAR_W + (bars - 1) * WAVE_GAP
    const left = totalW < w ? Math.max(0, Math.floor((w - totalW) / 2)) : 0
    const scrollLeft = scrollRef.current?.scrollLeft ?? 0
    const absX = scrollLeft + x
    const rel = (absX - left) / Math.max(1, totalW)
    return Math.max(0, Math.min(1, rel))
  }

  function onWaveScroll() {
    // Redraw the visible segment while scrolling.
    if (scrollRafRef.current != null) return
    scrollRafRef.current = window.requestAnimationFrame(() => {
      scrollRafRef.current = null
      drawWave({ canvas: p.canvasRef.current, peaks: p.peaks, frac, scrollLeft: scrollRef.current?.scrollLeft ?? 0 })
    })
  }

  function onPointerDown(e: React.PointerEvent) {
    if (!p.src) return
    const f = fracFromClientX(e.clientX)
    if (f == null) return
    ;(e.currentTarget as any).setPointerCapture?.(e.pointerId)
    seekToFrac(f)
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!p.src) return
    if ((e.buttons & 1) === 0) return
    const f = fracFromClientX(e.clientX)
    if (f == null) return
    seekToFrac(f)
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (!p.src) return
    const a = audioRef.current
    if (!a) return
    const d = a.duration
    if (!Number.isFinite(d) || d <= 0) return
    const step = 2 // seconds
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      a.currentTime = Math.max(0, a.currentTime - step)
      setCurrentTime(a.currentTime)
    }
    if (e.key === 'ArrowRight') {
      e.preventDefault()
      a.currentTime = Math.min(d, a.currentTime + step)
      setCurrentTime(a.currentTime)
    }
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault()
      togglePlay()
    }
  }

  function Icon(p2: { name: 'play' | 'pause' | 'skipBack' | 'skipFwd' | 'volume' | 'muted' | 'reset' | 'download' }) {
    const common = { className: 'wfIcon', viewBox: '0 0 24 24', 'aria-hidden': true as const }
    switch (p2.name) {
      case 'play':
        return (
          <svg {...common}>
            <path fill="currentColor" stroke="none" d="M8 5v14l11-7z" />
          </svg>
        )
      case 'pause':
        return (
          <svg {...common}>
            <path fill="currentColor" stroke="none" d="M6 5h4v14H6zM14 5h4v14h-4z" />
          </svg>
        )
      case 'skipBack':
        return (
          <svg {...common}>
            <path fill="currentColor" stroke="none" d="M6 6h2v12H6z" />
            <path fill="currentColor" stroke="none" d="M18 6L8 12l10 6z" />
          </svg>
        )
      case 'skipFwd':
        return (
          <svg {...common}>
            <path fill="currentColor" stroke="none" d="M16 6h2v12h-2z" />
            <path fill="currentColor" stroke="none" d="M6 6l10 6-10 6z" />
          </svg>
        )
      case 'volume':
        return (
          <svg {...common}>
            <path d="M11 5L7 9H4v6h3l4 4z" />
            <path d="M15 9a4 4 0 0 1 0 6" />
          </svg>
        )
      case 'muted':
        return (
          <svg {...common}>
            <path d="M11 5L7 9H4v6h3l4 4z" />
            <path d="M16 9l4 6" />
            <path d="M20 9l-4 6" />
          </svg>
        )
      case 'reset':
        return (
          <svg {...common}>
            <path d="M21 2v6h-6" />
            <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
            <path d="M3 22v-6h6" />
            <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
          </svg>
        )
      case 'download':
        return (
          <svg {...common}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <path d="M7 10l5 5 5-5" />
            <path d="M12 15V3" />
          </svg>
        )
    }
  }

  const speedLabel = speed === 1 ? '1x' : `${speed}x`

  const waveContentW = useMemo(() => {
    if (!p.peaks || p.peaks.length === 0) return 0
    return Math.max(1, p.peaks.length * WAVE_STEP - WAVE_GAP)
  }, [p.peaks])

  return (
    <div className="wfPlayer">
      <div
        className="waveform wfWave"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        tabIndex={0}
        onKeyDown={onKeyDown}
        aria-label={`${p.label} waveform`}
      >
        <div
          className="waveformScroll"
          ref={scrollRef}
          onScroll={onWaveScroll}
          style={{ ['--wave-content-w' as any]: waveContentW ? `${waveContentW}px` : '100%' }}
        >
          <canvas ref={p.canvasRef as any} className="waveformCanvas waveformCanvasSticky" />
        </div>
        {(() => {
          const overlay = p.overlayLabel ?? (p.isDecoding ? 'Rendering waveform' : p.src ? null : 'No audio')
          if (!overlay) return null

          return (
            <div className="waveformOverlay muted small">
              {overlay.startsWith('Generating') ? (
                <span className="wfOverlayRow">
                  <span className="wfSpinner" aria-hidden={true} />
                  <span>{overlay}</span>
                </span>
              ) : (
                <span>{overlay}</span>
              )}
            </div>
          )
        })()}
      </div>

      <div className="wfTimes" aria-label="Time">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      <div className="wfControls">
        <div className="wfLeft">
          <div className="wfVolumeWrap">
            <button
              type="button"
              className="wfBtn"
              onClick={() => setShowVolume((v) => !v)}
              aria-label="Volume"
              aria-expanded={showVolume}
              title="Volume"
            >
              <Icon name={isMuted ? 'muted' : 'volume'} />
            </button>
            {showVolume ? (
              <div className="wfVolumePopover" role="group" aria-label="Volume slider">
                <button type="button" className="wfVolMute" onClick={toggleMute} aria-label={isMuted ? 'Unmute' : 'Mute'}>
                  <Icon name={isMuted ? 'muted' : 'volume'} />
                </button>
                <input
                  className="wfVolRange"
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={isMuted ? 0 : volume}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    setVolume(v)
                    if (v > 0) setIsMuted(false)
                  }}
                  disabled={!p.src}
                  aria-label="Volume"
                />
              </div>
            ) : null}
          </div>

          <button type="button" className="wfPill" onClick={cycleSpeed} disabled={!p.src} aria-label="Speed" title="Speed">
            {speedLabel}
          </button>
        </div>

        <div className="wfCenter">
          <button type="button" className="wfBtn" onClick={() => skipBySeconds(-2)} disabled={!p.src} aria-label="Back 2 seconds" title="Back 2s">
            <Icon name="skipBack" />
          </button>
          <button
            type="button"
            className="wfBtn wfBtnPrimary"
            onClick={togglePlay}
            disabled={!p.src}
            aria-label={isPlaying ? 'Pause' : 'Play'}
            aria-pressed={isPlaying}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            <Icon name={isPlaying ? 'pause' : 'play'} />
          </button>
          <button type="button" className="wfBtn" onClick={() => skipBySeconds(2)} disabled={!p.src} aria-label="Forward 2 seconds" title="Forward 2s">
            <Icon name="skipFwd" />
          </button>
        </div>

        <div className="wfRight">
          {p.onSave ? (
            <button
              type="button"
              className="wfBtn"
              onClick={p.onSave}
              disabled={!p.src || p.isSaving}
              aria-label={p.isSaving ? 'Saving output audio' : 'Save output audio'}
              title={p.isSaving ? 'Saving…' : 'Save'}
            >
              {p.isSaving ? <span className="wfSpinner" aria-hidden={true} /> : <Icon name="download" />}
            </button>
          ) : null}
          {p.onClear ? (
            <button type="button" className="wfBtn" onClick={p.onClear} aria-label="Clear" title="Clear">
              <Icon name="reset" />
            </button>
          ) : null}
        </div>
      </div>

      <audio ref={audioRef} className="audioEl" src={p.src ?? undefined} preload="metadata" />
    </div>
  )
}

export function WorkspaceScreen(props: {
  deviceSpec: string
  referenceAudioName: string | null
  referenceText: string
  targetText: string
  cfgValue: number
  inferenceSteps: number
  progress: ProgressEventPayload | null
  audioUrl: string | null
  isGenerating: boolean
  onPickReferenceAudioBytes: (payload: { name: string; bytes: Uint8Array }) => void
  onClearReferenceAudio: () => void
  onChangeReferenceText: (v: string) => void
  onChangeTargetText: (v: string) => void
  onChangeCfgValue: (v: number) => void
  onChangeInferenceSteps: (v: number) => void
  onGenerate: () => void
  onStop: () => void
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const refWaveCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const outWaveCanvasRef = useRef<HTMLCanvasElement | null>(null)

  const [refAudioUrl, setRefAudioUrl] = useState<string | null>(null)
  const refAudioUrlRef = useRef<string | null>(null)
  const [refWavePeaks, setRefWavePeaks] = useState<Float32Array | null>(null)
  const [refAudioError, setRefAudioError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState<boolean>(false)
  const [isDecoding, setIsDecoding] = useState<boolean>(false)
  const [waveRedrawTick, setWaveRedrawTick] = useState<number>(0)

  const [outWavePeaks, setOutWavePeaks] = useState<Float32Array | null>(null)
  const [outWaveError, setOutWaveError] = useState<string | null>(null)
  const [isOutDecoding, setIsOutDecoding] = useState<boolean>(false)

  const [isSavingOut, setIsSavingOut] = useState<boolean>(false)

  function useAutoGrowTextarea(value: string, maxRows: number) {
    const ref = useRef<HTMLTextAreaElement | null>(null)

    useLayoutEffect(() => {
      const el = ref.current
      if (!el) return

      // Reset height so it can shrink as well.
      el.style.height = '0px'

      // Determine a reasonable max height from line-height.
      const cs = window.getComputedStyle(el)
      const lineH = Number.parseFloat(cs.lineHeight || '0')
      const padTop = Number.parseFloat(cs.paddingTop || '0')
      const padBot = Number.parseFloat(cs.paddingBottom || '0')
      const maxH = lineH > 0 ? lineH * maxRows + padTop + padBot : Number.POSITIVE_INFINITY

      const next = el.scrollHeight
      el.style.height = `${Math.min(next, maxH)}px`
      el.style.overflowY = next > maxH ? 'auto' : 'hidden'
    }, [value, maxRows])

    return ref
  }

  function isAudioFile(f: File) {
    const name = f.name.toLowerCase()
    const m = /\.([a-z0-9]+)$/.exec(name)
    const ext = m?.[1] ?? ''
    const extOk = (SUPPORTED_REF_AUDIO_EXTS as readonly string[]).includes(ext)

    const type = (f.type || '').toLowerCase()
    const typeOk = type === '' || SUPPORTED_REF_AUDIO_MIMES.has(type)

    return extOk && typeOk
  }

  async function computeWavePeaks(arrayBuffer: ArrayBuffer): Promise<Float32Array> {
    const Ctx = (window as any).AudioContext || (window as any).webkitAudioContext
    if (!Ctx) throw new Error('AudioContext not available')
    const ctx = new Ctx()
    try {
      // Some engines require an explicit resume in user-gesture flows.
      await ctx.resume().catch(() => {})
      const audioBuffer: AudioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0))
      const durationSec = audioBuffer.duration
      const bars = Math.max(
        WAVE_MIN_BARS,
        Math.min(WAVE_MAX_BARS, Math.round(Math.max(0.01, durationSec) * WAVE_BARS_PER_SEC))
      )
      const ch0 = audioBuffer.getChannelData(0)
      const step = Math.max(1, Math.floor(ch0.length / bars))
      const peaks = new Float32Array(bars)
      for (let i = 0; i < bars; i++) {
        const start = i * step
        const end = Math.min(ch0.length, start + step)
        let max = 0
        for (let j = start; j < end; j++) {
          const v = Math.abs(ch0[j])
          if (v > max) max = v
        }
        peaks[i] = max
      }
      return peaks
    } finally {
      await ctx.close().catch(() => {})
    }
  }

  async function handlePickReferenceAudio(file: File) {
    if (!isAudioFile(file)) {
      setRefAudioError(`Unsupported reference audio. Supported extensions: ${SUPPORTED_REF_AUDIO_EXT_LIST}.`)
      return
    }

    setRefAudioError(null)
    setIsDecoding(true)

    // Revoke any previous object URL.
    if (refAudioUrlRef.current) {
      URL.revokeObjectURL(refAudioUrlRef.current)
      refAudioUrlRef.current = null
    }

    const url = URL.createObjectURL(file)
    refAudioUrlRef.current = url
    setRefAudioUrl(url)

    const buf = await file.arrayBuffer()
    const bytes = new Uint8Array(buf)
    props.onPickReferenceAudioBytes({ name: file.name, bytes })

    try {
      const peaks = await computeWavePeaks(buf)
      setRefWavePeaks(peaks)
    } catch (e) {
      // Preview should be best-effort; keep playback even if waveform fails.
      setRefWavePeaks(null)
      setRefAudioError(`Waveform preview failed (file still loaded): ${String(e)}`)
    } finally {
      setIsDecoding(false)
    }
  }

  async function handlePickReferenceAudioPath(path: string) {
    // Native dialog already filters, but keep a cheap guard for safety.
    const name = path.split('/').pop() ?? path
    if (!isSupportedRefAudioName(name)) {
      setRefAudioError(`Unsupported reference audio. Supported extensions: ${SUPPORTED_REF_AUDIO_EXT_LIST}.`)
      return
    }

    setRefAudioError(null)
    try {
      const bytes = await readFile(path)
      // Construct a File so we can reuse the existing preview + waveform path.
      const f = new File([bytes], name)
      await handlePickReferenceAudio(f)
    } catch (e) {
      setRefAudioError(`Failed to read reference audio: ${String(e)}`)
    }
  }

  async function onSelectReferenceAudio() {
    // Use Tauri native dialog so the picker can actually filter file types.
    const picked = await open({
      multiple: false,
      filters: [{ name: 'Audio', extensions: SUPPORTED_REF_AUDIO_EXTS_DIALOG }],
    })
    if (!picked) return
    if (Array.isArray(picked)) return
    await handlePickReferenceAudioPath(picked)
  }

  function clearReferenceAudio() {
    props.onClearReferenceAudio()
    setRefWavePeaks(null)
    setRefAudioError(null)
    setIsDecoding(false)
    if (refAudioUrlRef.current) {
      URL.revokeObjectURL(refAudioUrlRef.current)
      refAudioUrlRef.current = null
    }
    setRefAudioUrl(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  useEffect(() => {
    return () => {
      if (refAudioUrlRef.current) {
        URL.revokeObjectURL(refAudioUrlRef.current)
        refAudioUrlRef.current = null
      }
    }
  }, [])

  const generatedSec = useMemo(() => {
    const ms = props.progress?.progress?.generated_ms
    if (ms == null) return null
    return ms / 1000
  }, [props.progress])

  const generatedSecForOverlay = useMemo(() => {
    if (!props.isGenerating) return null
    return generatedSec ?? 0
  }, [generatedSec, props.isGenerating])

  const requireRefText = props.referenceAudioName != null
  const refTextOk = props.referenceText.trim().length > 0
  const canGenerate = props.targetText.trim().length > 0 && (!requireRefText || refTextOk) && !props.isGenerating

  const referenceTextRef = useAutoGrowTextarea(props.referenceText, 4)
  const targetTextRef = useAutoGrowTextarea(props.targetText, 4)

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

  useEffect(() => {
    // Prevent the webview from navigating/opening the file when dropped.
    function onDragOver(e: DragEvent) {
      e.preventDefault()
    }
    function onDrop(e: DragEvent) {
      e.preventDefault()
    }
    window.addEventListener('dragover', onDragOver, true)
    window.addEventListener('drop', onDrop, true)
    return () => {
      window.removeEventListener('dragover', onDragOver, true)
      window.removeEventListener('drop', onDrop, true)
    }
  }, [])

  useEffect(() => {
    function onResize() {
      setWaveRedrawTick((t) => t + 1)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  useEffect(() => {
    // Redraw waveforms on resize.
    drawWave({ canvas: refWaveCanvasRef.current, peaks: refWavePeaks, frac: 0 })
    drawWave({ canvas: outWaveCanvasRef.current, peaks: outWavePeaks, frac: 0 })
  }, [refWavePeaks, outWavePeaks, waveRedrawTick])

  useEffect(() => {
    // Best-effort waveform for output audio.
    const src = props.audioUrl
    if (!src) {
      setOutWavePeaks(null)
      setOutWaveError(null)
      setIsOutDecoding(false)
      return
    }
    let cancelled = false
    setIsOutDecoding(true)
    setOutWaveError(null)
    ;(async () => {
      try {
        const res = await fetch(src)
        const buf = await res.arrayBuffer()
        const peaks = await computeWavePeaks(buf)
        if (cancelled) return
        setOutWavePeaks(peaks)
      } catch (e) {
        if (cancelled) return
        setOutWavePeaks(null)
        setOutWaveError(String(e))
      } finally {
        if (cancelled) return
        setIsOutDecoding(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [props.audioUrl])

  function makeDefaultOutFileName() {
    const d = new Date()
    const pad2 = (n: number) => String(n).padStart(2, '0')
    const stamp = `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(
      d.getMinutes(),
    )}${pad2(d.getSeconds())}`
    return `voxcpm-${stamp}.wav`
  }

  async function onSaveOutput() {
    if (!props.audioUrl) return
    setIsSavingOut(true)
    try {
      const path = await save({
        defaultPath: makeDefaultOutFileName(),
        filters: [{ name: 'WAV audio', extensions: ['wav'] }],
      })
      if (!path) return

      // audioUrl is a blob: URL created from the raw wav bytes in App.tsx.
      const res = await fetch(props.audioUrl)
      const buf = await res.arrayBuffer()
      await writeFile(path, new Uint8Array(buf))
    } catch (e) {
      console.error('[voxcpm] save output failed', e)
    } finally {
      setIsSavingOut(false)
    }
  }

  return (
    <div className="container workspaceScreen" aria-busy={props.isGenerating}>
      <div className="grid2">
        <div className="refTargetGrid">
          <div className="card">
            <div className="cardHeader">
              <div>
                <div className="h2">Reference</div>
                <div className="muted">Optional voice prompt</div>
              </div>
              <div className="chip">Device: {props.deviceSpec}</div>
            </div>
            <div className="cardBody">
              <label className="label" htmlFor="referenceAudio">
                Reference audio
              </label>
              <input
                id="referenceAudio"
                ref={fileInputRef}
                className="fileInputHidden"
                type="file"
                accept={REFERENCE_AUDIO_ACCEPT}
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (!f) return
                  void handlePickReferenceAudio(f)
                }}
              />

              <div
                className={
                  'dropzone' +
                  (props.referenceAudioName ? '' : ' dropzoneClickable') +
                  (isDragging ? ' dropzoneDragging' : '') +
                  (props.referenceAudioName ? ' dropzoneLoaded' : '') +
                  (refAudioError ? ' dropzoneError' : '')
                }
                role={props.referenceAudioName ? undefined : 'button'}
                tabIndex={props.referenceAudioName ? undefined : 0}
                aria-label={props.referenceAudioName ? 'Reference waveform' : 'Upload reference audio'}
                onClick={() => {
                  if (props.referenceAudioName) return
                  void onSelectReferenceAudio().catch((e) => {
                    // Fallback to the HTML file input if native dialog fails.
                    console.error('[voxcpm] reference audio dialog failed', e)
                    fileInputRef.current?.click()
                  })
                }}
                onKeyDown={(e) => {
                  if (props.referenceAudioName) return
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    void onSelectReferenceAudio().catch((err) => {
                      console.error('[voxcpm] reference audio dialog failed', err)
                      fileInputRef.current?.click()
                    })
                  }
                }}
                onDragEnter={(e) => {
                  if (props.referenceAudioName) return
                  e.preventDefault()
                  setIsDragging(true)
                }}
                onDragOver={(e) => {
                  if (props.referenceAudioName) return
                  e.preventDefault()
                  e.dataTransfer.dropEffect = 'copy'
                  setIsDragging(true)
                }}
                onDragLeave={(e) => {
                  if (props.referenceAudioName) return
                  e.preventDefault()
                  setIsDragging(false)
                }}
                onDrop={(e) => {
                  if (props.referenceAudioName) return
                  e.preventDefault()
                  setIsDragging(false)
                  const f = e.dataTransfer.files?.[0]
                  if (!f) return
                  void handlePickReferenceAudio(f)
                }}
              >
                {props.referenceAudioName == null ? (
                  <div>
                    <div className="dropzoneTitle">
                      {isDragging ? 'Drop to upload' : 'Drop reference audio here'}
                    </div>
                    <div className="dropzoneHint muted small">or click to select</div>
                  </div>
                ) : (
                  <WaveformPlayer
                    label="Reference"
                    src={refAudioUrl}
                    peaks={refWavePeaks}
                    canvasRef={refWaveCanvasRef}
                    isDecoding={isDecoding}
                    waveRedrawTick={waveRedrawTick}
                    onClear={clearReferenceAudio}
                  />
                )}
              </div>

              {refAudioError ? <div className="error">{refAudioError}</div> : null}

              <div className="field">
                <label className="label" htmlFor="referenceText">
                  Reference text {requireRefText ? '(required)' : ''}
                </label>
                <textarea
                  id="referenceText"
                  ref={referenceTextRef}
                  className="input inputWrap"
                  rows={1}
                  placeholder="transcript of reference audio ..."
                  value={props.referenceText}
                  onChange={(e) => props.onChangeReferenceText(e.target.value)}
                  onKeyDown={(e) => {
                    // Keep the "single-line" feel: do not allow newline insertion.
                    if (e.key === 'Enter' && !(e.nativeEvent as any).isComposing) {
                      e.preventDefault()
                    }
                  }}
                />
                {requireRefText && !refTextOk ? (
                  <div className="error">Reference text is required when reference audio is provided.</div>
                ) : null}
              </div>
            </div>
          </div>

          <div className="card targetCard">
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
                  ref={targetTextRef}
                  className="input inputWrap"
                  rows={1}
                  value={props.targetText}
                  onChange={(e) => props.onChangeTargetText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !(e.nativeEvent as any).isComposing) {
                      e.preventDefault()
                    }
                  }}
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
              <button
                className="btn btnPrimary"
                onClick={props.onGenerate}
                disabled={!canGenerate}
                title="Cmd/Ctrl+Enter"
                aria-keyshortcuts="Control+Enter Meta+Enter"
              >
                {props.isGenerating ? 'Generating…' : 'Generate'}
              </button>
              <button
                className="btn btnDanger"
                onClick={props.onStop}
                disabled={!props.isGenerating}
                title="Esc"
                aria-keyshortcuts="Escape"
              >
                Stop
              </button>
            </div>
          </div>
        </div>

        <div className="card outputCard">
          <div className="cardHeader">
            <div>
              <div className="h2">Output</div>
              <div className="muted">Generated audio</div>
            </div>
          </div>
          <div className="cardBody">
            <WaveformPlayer
              label="Output"
              src={props.audioUrl}
              peaks={outWavePeaks}
              canvasRef={outWaveCanvasRef}
              isDecoding={props.isGenerating || isOutDecoding}
              waveRedrawTick={waveRedrawTick}
              overlayLabel={props.isGenerating ? `Generating ${formatTime(generatedSecForOverlay)}` : null}
              onSave={() => void onSaveOutput()}
              isSaving={isSavingOut}
            />
            {outWaveError ? <div className="error">Waveform failed: {outWaveError}</div> : null}

            {/* Log intentionally hidden in Workspace UI (still collected in App state). */}
          </div>
        </div>
      </div>
    </div>
  )
}
