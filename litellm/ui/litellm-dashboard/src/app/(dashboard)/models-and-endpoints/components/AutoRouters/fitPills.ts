/**
 * How many pills fit on ONE row of a given width, leaving room for a "+N" counter.
 *
 * jsdom reports no layout and the shared ResizeObserver mock only fires inside chart
 * subtrees, so this stays a pure width-in / count-out function: the component measures and
 * this decides, which keeps the overflow rule unit-testable.
 */

const CHAR_WIDTH = 6.5;
const PILL_PADDING = 18;
const PILL_GAP = 4;
const OVERFLOW_WIDTH = 28;

export const pillWidth = (label: string): number => label.length * CHAR_WIDTH + PILL_PADDING;

export interface FittedPills {
  visible: string[];
  overflow: number;
}

export const fitPills = (labels: string[], availableWidth: number): FittedPills => {
  if (labels.length === 0) return { visible: [], overflow: 0 };

  // Unmeasured (0 or negative) means the first paint before ResizeObserver reports. Show one
  // pill rather than all of them, so the row never flashes multi-line and then collapses.
  if (availableWidth <= 0) {
    return { visible: labels.slice(0, 1), overflow: labels.length - 1 };
  }

  const fitted: string[] = [];
  let used = 0;

  for (const [index, label] of labels.entries()) {
    const remaining = labels.length - index - 1;
    const gap = fitted.length === 0 ? 0 : PILL_GAP;
    // Anything still queued after this pill needs room for the "+N" counter beside it.
    const reserve = remaining > 0 ? PILL_GAP + OVERFLOW_WIDTH : 0;

    if (used + gap + pillWidth(label) + reserve > availableWidth) break;

    used += gap + pillWidth(label);
    fitted.push(label);
  }

  // Always show at least one pill; a single over-long name truncates via CSS instead of
  // collapsing the cell to a bare "+N".
  if (fitted.length === 0) {
    return { visible: labels.slice(0, 1), overflow: labels.length - 1 };
  }

  return { visible: fitted, overflow: labels.length - fitted.length };
};
