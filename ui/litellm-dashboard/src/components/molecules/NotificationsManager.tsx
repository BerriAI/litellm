import { notification } from "antd"
import { parseErrorMessage } from "../shared/errorUtils"

type Placement = "top" | "topLeft" | "topRight" | "bottom" | "bottomLeft" | "bottomRight"

type NotificationConfig = {
  message?: string
  description?: string
  duration?: number
  placement?: Placement
  key?: string
}

type NotificationConfigResolved = Omit<NotificationConfig, "message"> & { message: string }

function defaultPlacement(): Placement {
  return "topRight"
}

function normalize(input: string | NotificationConfig, fallbackTitle: string): NotificationConfigResolved {
  if (typeof input === "string") return { message: fallbackTitle, description: input }
  return { message: input.message ?? fallbackTitle, ...input }
}

function toIntMaybe(val: any): number | undefined {
  if (typeof val === "number") return val
  if (typeof val === "string" && /^\d+$/.test(val)) return parseInt(val, 10)
  return undefined
}

function titleFor(status?: number, desc?: string): string {
  const d = (desc || "").toLowerCase()
  if (status && status >= 500) return "Server Error"
  if (status === 401 || d.includes("authentication")) return "Authentication Error"
  if (status === 403 || d.includes("forbidden") || d.includes("access denied")) return "Access Denied"
  if (status === 404 || d.includes("not found")) return "Not Found"
  if (status === 429 || d.includes("rate limit") || d.includes("tpm") || d.includes("rpm")) return "Rate Limit Exceeded"
  if (d.includes("enterprise") || d.includes("premium")) return "Info"
  if (status && status >= 400) return "Request Error"
  return "Error"
}

const NotificationManager = {
  error(input: string | NotificationConfig) {
    const cfg = normalize(input, "Error")
    notification.error({
      ...cfg,
      placement: cfg.placement ?? defaultPlacement(),
      duration: cfg.duration ?? 6,
    })
  },

  warning(input: string | NotificationConfig) {
    const cfg = normalize(input, "Warning")
    notification.warning({
      ...cfg,
      placement: cfg.placement ?? defaultPlacement(),
      duration: cfg.duration ?? 5,
    })
  },

  info(input: string | NotificationConfig) {
    const cfg = normalize(input, "Info")
    notification.info({
      ...cfg,
      placement: cfg.placement ?? defaultPlacement(),
      duration: cfg.duration ?? 4,
    })
  },

  success(input: string | NotificationConfig) {
    const cfg = normalize(input, "Success")
    notification.success({
      ...cfg,
      placement: cfg.placement ?? defaultPlacement(),
      duration: cfg.duration ?? 3.5,
    })
  },

  // Show a categorized notification from a backend-style error
  fromBackendError(error: any, extra?: Omit<NotificationConfig, "message" | "description">) {
    const status = toIntMaybe(error?.response?.status) ?? toIntMaybe(error?.status_code) ?? toIntMaybe(error?.code)

    // Try to surface the most informative backend message
    const backendMsg =
      error?.response?.data?.error?.message ??
      error?.response?.data?.message ??
      error?.response?.data?.error ??
      error?.detail ??
      error?.message ??
      error

    const description = parseErrorMessage(backendMsg)
    const title = titleFor(status, description)

    const base = {
      ...(extra ?? {}),
      message: title,
      description,
      placement: extra?.placement ?? defaultPlacement(),
    }

    if (title === "Rate Limit Exceeded" || title === "Info") {
      notification.warning({ ...base, duration: extra?.duration ?? 7 })
      return
    }
    if (title === "Server Error") {
      notification.error({ ...base, duration: extra?.duration ?? 8 })
      return
    }
    if (
      title === "Request Error" ||
      title === "Authentication Error" ||
      title === "Access Denied" ||
      title === "Not Found" ||
      title === "Error"
    ) {
      notification.error({ ...base, duration: extra?.duration ?? 6 })
      return
    }
    notification.info({ ...base, duration: extra?.duration ?? 4 })
  },

  clear() {
    notification.destroy()
  },
}

export default NotificationManager
