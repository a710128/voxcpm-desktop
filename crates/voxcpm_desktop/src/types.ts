export type DeviceSpec = string

export type VoxcpmStage = 'download' | 'verify' | 'load' | 'ready' | 'error'

export type CapabilitiesResponse = {
  devices: DeviceSpec[]
  mirrorDefault: boolean
  defaultModel: {
    repoId: string
    revision: string
  }
}

export type StageEventPayload = {
  stage: VoxcpmStage | string
  message?: string
}

export type DownloadEventPayload =
  | {
      stage: 'downloading'
      file: string
      done: number
      total: number
      bytesDownloaded: number
      bytesTotal: number | null
      percent: number | null
    }
  | {
      stage: 'done'
      model_dir: string
    }

export type ProgressEventPayload = {
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
