const KEY_MIRROR_DEFAULT = 'voxcpm:mig:mirrorDefault'

export function getMirrorDefault(): boolean {
  const v = localStorage.getItem(KEY_MIRROR_DEFAULT)
  if (v == null) return false
  return v === '1' || v.toLowerCase() === 'true'
}

export function setMirrorDefault(v: boolean) {
  localStorage.setItem(KEY_MIRROR_DEFAULT, v ? '1' : '0')
}
