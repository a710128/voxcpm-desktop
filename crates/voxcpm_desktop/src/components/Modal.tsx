import { useEffect, useId, useRef } from 'react'

import type { ReactNode } from 'react'

export function Modal(props: {
  title: string
  children: ReactNode
  footer?: ReactNode
  onClose?: () => void
}) {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const closeBtnRef = useRef<HTMLButtonElement | null>(null)
  const onCloseRef = useRef<(() => void) | undefined>(props.onClose)

  useEffect(() => {
    onCloseRef.current = props.onClose
  }, [props.onClose])

  useEffect(() => {
    // Best-effort focus management (no full focus trap).
    const el =
      closeBtnRef.current ??
      dialogRef.current?.querySelector<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
    el?.focus()
  }, [])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== 'Escape') return
      if (!onCloseRef.current) return
      e.preventDefault()
      onCloseRef.current()
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  return (
    <div className="modalOverlay" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <div className="modal" ref={dialogRef} tabIndex={-1}>
        <div className="modalHeader">
          <div className="modalTitle" id={titleId}>
            {props.title}
          </div>
          {props.onClose ? (
            <button
              ref={closeBtnRef}
              className="btn btnGhost"
              onClick={props.onClose}
              aria-label="Close"
            >
              X
            </button>
          ) : null}
        </div>
        <div className="modalBody">{props.children}</div>
        {props.footer ? <div className="modalFooter">{props.footer}</div> : null}
      </div>
    </div>
  )
}
