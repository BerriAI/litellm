export const SCROLL_PIN_THRESHOLD_PX = 48;

export function isPinnedToBottom(
  metrics: { scrollTop: number; scrollHeight: number; clientHeight: number },
  thresholdPx = SCROLL_PIN_THRESHOLD_PX,
): boolean {
  return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight <= thresholdPx;
}
