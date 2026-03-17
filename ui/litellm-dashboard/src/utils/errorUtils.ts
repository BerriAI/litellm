/**
 * Shared error-message extraction utility.
 *
 * Handles the common shapes returned by LiteLLM / FastAPI:
 *   - Error instances (err.message)
 *   - { detail: "string" }
 *   - { detail: [{ msg, loc, type }] }   (FastAPI 422)
 *   - { detail: { error: "string" } }
 *   - { message: "string" }
 *   - anything else → JSON.stringify / String()
 */
export function extractErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (err && typeof err === "object") {
    const e = err as Record<string, unknown>;
    const detail = e.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((d: unknown) => {
        if (d && typeof d === "object") {
          const item = d as Record<string, unknown>;
          return typeof item.msg === "string" ? item.msg : JSON.stringify(d);
        }
        return String(d);
      }).join("; ");
    }
    if (detail && typeof detail === "object") {
      const detailObj = detail as Record<string, unknown>;
      if (typeof detailObj.error === "string") return detailObj.error;
    }
    if (typeof e.message === "string") return e.message;
    return JSON.stringify(err);
  }
  return String(err);
}
