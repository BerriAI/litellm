export interface ScatterLabelPoint {
  id: string;
  text: string;
  centerX: number;
  centerY: number;
  dotRadius: number;
  preferRight: boolean;
}

export interface ScatterLabelPlotArea {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ScatterLabelBounds {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export interface ScatterLabelPlacement {
  id: string;
  text: string;
  x: number;
  y: number;
  textAnchor: "start" | "end";
  bounds: ScatterLabelBounds;
}

const LABEL_WIDTH_PER_CHARACTER = 7;
const LABEL_TOP_FROM_BASELINE = 9;
const LABEL_BOTTOM_FROM_BASELINE = 3;
const LABEL_DOT_GAP = 4;
const COLLISION_GAP = 2;
const VERTICAL_NUDGE = 14;

const labelBounds = (
  x: number,
  y: number,
  text: string,
  textAnchor: "start" | "end",
): ScatterLabelBounds => {
  const width = text.length * LABEL_WIDTH_PER_CHARACTER;
  const left = textAnchor === "start" ? x : x - width;
  return {
    left,
    right: left + width,
    top: y - LABEL_TOP_FROM_BASELINE,
    bottom: y + LABEL_BOTTOM_FROM_BASELINE,
  };
};

const overlaps = (left: ScatterLabelBounds, right: ScatterLabelBounds): boolean => {
  const overlapsHorizontally =
    left.left < right.right + COLLISION_GAP && left.right + COLLISION_GAP > right.left;
  const overlapsVertically =
    left.top < right.bottom + COLLISION_GAP && left.bottom + COLLISION_GAP > right.top;
  return overlapsHorizontally && overlapsVertically;
};

const isInsidePlot = (bounds: ScatterLabelBounds, plotArea: ScatterLabelPlotArea): boolean => {
  const insideHorizontalBounds = bounds.left >= plotArea.x && bounds.right <= plotArea.x + plotArea.width;
  const insideVerticalBounds = bounds.top >= plotArea.y && bounds.bottom <= plotArea.y + plotArea.height;
  return insideHorizontalBounds && insideVerticalBounds;
};

const labelCandidates = (point: ScatterLabelPoint, plotArea: ScatterLabelPlotArea): ScatterLabelPlacement[] => {
  const sideOrder = point.preferRight ? (["start", "end"] as const) : (["end", "start"] as const);
  const offsetCount = Math.ceil(plotArea.height / VERTICAL_NUDGE) + 1;
  const verticalOffsets = Array.from({ length: offsetCount * 2 - 1 }, (_, index) =>
    index === 0 ? 0 : Math.ceil(index / 2) * VERTICAL_NUDGE * (index % 2 === 1 ? -1 : 1),
  );
  return sideOrder.flatMap((textAnchor) =>
    verticalOffsets.map((offset) => {
      const x =
        textAnchor === "start"
          ? point.centerX + point.dotRadius + LABEL_DOT_GAP
          : point.centerX - point.dotRadius - LABEL_DOT_GAP;
      const y = point.centerY + 3 + offset;
      return {
        id: point.id,
        text: point.text,
        x,
        y,
        textAnchor,
        bounds: labelBounds(x, y, point.text, textAnchor),
      };
    }),
  );
};

export const placeScatterLabels = (
  points: readonly ScatterLabelPoint[],
  plotArea: ScatterLabelPlotArea,
): ScatterLabelPlacement[] =>
  points.reduce<ScatterLabelPlacement[]>((placed, point) => {
    const candidate = labelCandidates(point, plotArea).find(
      (label) =>
        isInsidePlot(label.bounds, plotArea) &&
        points.every((dot) => {
          const dotBounds = {
            left: dot.centerX - dot.dotRadius,
            right: dot.centerX + dot.dotRadius,
            top: dot.centerY - dot.dotRadius,
            bottom: dot.centerY + dot.dotRadius,
          };
          return !overlaps(label.bounds, dotBounds);
        }) &&
        placed.every((previous) => !overlaps(label.bounds, previous.bounds)),
    );
    return candidate ? [...placed, candidate] : placed;
  }, []);
