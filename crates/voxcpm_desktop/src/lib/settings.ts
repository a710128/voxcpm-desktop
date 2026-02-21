const KEY_MIRROR_DEFAULT_LEGACY = 'voxcpm:mig:mirrorDefault'

const KEY_LAUNCH_MIRROR = 'voxcpm:launch:mirror'
const KEY_LAUNCH_DEVICE_SPEC = 'voxcpm:launch:deviceSpec'
// Legacy: empty string means "use default repo".
const KEY_LAUNCH_REPO_ID = 'voxcpm:launch:repoId'

const KEY_LAUNCH_MODEL_SOURCE = 'voxcpm:launch:modelSource'
const KEY_LAUNCH_CUSTOM_REPO_ID = 'voxcpm:launch:customRepoId'

export type LaunchModelSource = 'default' | 'custom'

function parseBool(v: string | null): boolean | null {
  if (v == null) return null
  if (v === '1') return true
  if (v === '0') return false
  const s = v.toLowerCase()
  if (s === 'true') return true
  if (s === 'false') return false
  return null
}

export function getLaunchMirror(): boolean {
  // Prefer new key; fall back to legacy mirrorDefault.
  const v = parseBool(localStorage.getItem(KEY_LAUNCH_MIRROR))
  if (v != null) return v
  const legacy = parseBool(localStorage.getItem(KEY_MIRROR_DEFAULT_LEGACY))
  if (legacy != null) return legacy
  return false
}

export function setLaunchMirror(v: boolean) {
  localStorage.setItem(KEY_LAUNCH_MIRROR, v ? '1' : '0')
}

export function getLaunchDeviceSpec(): string | null {
  const v = localStorage.getItem(KEY_LAUNCH_DEVICE_SPEC)
  if (v == null) return null
  const s = v.trim()
  return s === '' ? null : s
}

export function setLaunchDeviceSpec(v: string) {
  localStorage.setItem(KEY_LAUNCH_DEVICE_SPEC, v)
}

export function getLaunchRepoId(): string | null {
  const v = localStorage.getItem(KEY_LAUNCH_REPO_ID)
  if (v == null) return null
  const s = v.trim()
  return s === '' ? null : s
}

export function setLaunchRepoId(v: string | null) {
  localStorage.setItem(KEY_LAUNCH_REPO_ID, (v ?? '').trim())
}

export function getLaunchModelSource(): LaunchModelSource {
  const raw = localStorage.getItem(KEY_LAUNCH_MODEL_SOURCE)
  if (raw === 'default' || raw === 'custom') return raw

  // Migrate from legacy repoId key.
  const legacyRepo = getLaunchRepoId()
  return legacyRepo ? 'custom' : 'default'
}

export function setLaunchModelSource(v: LaunchModelSource) {
  localStorage.setItem(KEY_LAUNCH_MODEL_SOURCE, v)
}

export function getLaunchCustomRepoId(): string {
  const raw = localStorage.getItem(KEY_LAUNCH_CUSTOM_REPO_ID)
  if (raw != null) {
    const s = raw.trim()
    if (s !== '') return s
  }

  // Migrate from legacy repoId key.
  return getLaunchRepoId() ?? ''
}

export function setLaunchCustomRepoId(v: string) {
  localStorage.setItem(KEY_LAUNCH_CUSTOM_REPO_ID, v.trim())
}
