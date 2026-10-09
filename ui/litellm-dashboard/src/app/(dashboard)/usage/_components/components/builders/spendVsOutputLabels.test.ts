import { describe, expect, it } from "vitest";
import { placeScatterLabels, type ScatterLabelBounds, type ScatterLabelPoint } from "./spendVsOutputLabels";

const boundsOverlap = (left: ScatterLabelBounds, right: ScatterLabelBounds): boolean => {
  const overlapsHorizontally = left.left < right.right && left.right > right.left;
  const overlapsVertically = left.top < right.bottom && left.bottom > right.top;
  return overlapsHorizontally && overlapsVertically;
};

describe("placeScatterLabels", () => {
  it("nudges bottom-left labels away from nearby dots and other labels", () => {
    const points: ScatterLabelPoint[] = [
      { id: "misbah", text: "Misbah", centerX: 136, centerY: 267, dotRadius: 5, preferRight: true },
      { id: "mik", text: "Mik", centerX: 105, centerY: 267, dotRadius: 5, preferRight: true },
      { id: "oliver", text: "Oliver", centerX: 115, centerY: 246, dotRadius: 5, preferRight: true },
    ];
    const plotArea = { x: 0, y: 0, width: 320, height: 280 };
    const labels = placeScatterLabels(points, plotArea);

    expect(labels).toHaveLength(points.length);
    labels.forEach((label) => {
      expect(label.bounds.left).toBeGreaterThanOrEqual(plotArea.x);
      expect(label.bounds.right).toBeLessThanOrEqual(plotArea.x + plotArea.width);
      expect(label.bounds.top).toBeGreaterThanOrEqual(plotArea.y);
      expect(label.bounds.bottom).toBeLessThanOrEqual(plotArea.y + plotArea.height);
      points.forEach((point) => {
        const dotBounds = {
          left: point.centerX - point.dotRadius,
          right: point.centerX + point.dotRadius,
          top: point.centerY - point.dotRadius,
          bottom: point.centerY + point.dotRadius,
        };
        expect(boundsOverlap(label.bounds, dotBounds)).toBe(false);
      });
    });
    labels.forEach((label, index) => {
      labels.slice(index + 1).forEach((next) => {
        expect(boundsOverlap(label.bounds, next.bounds)).toBe(false);
      });
    });

    const mikLabel = labels.find((label) => label.id === "mik");
    expect(mikLabel?.textAnchor).toBe("start");
    expect(mikLabel?.y).toBeLessThan(points[1].centerY);
  });
});
