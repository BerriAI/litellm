import { describe, expect, it } from "vitest";
import { fitViewport, graphSize, MAX_SCALE, MIN_SCALE, zoomViewport } from "./viewport";

describe("workflow viewport", () => {
  it.each([3, 120])("fits %i sequential steps and terminals inside the canvas", (count) => {
    const container = { width: 720, height: 640 };
    const graph = graphSize(count);
    const view = fitViewport(container, graph);
    expect(view.x).toBeGreaterThanOrEqual(0);
    expect(view.y).toBeGreaterThanOrEqual(0);
    expect(view.x + graph.width * view.scale).toBeLessThanOrEqual(container.width);
    expect(view.y + graph.height * view.scale).toBeLessThanOrEqual(container.height);
  });

  it("keeps the zoom anchor fixed while enforcing usable scale limits", () => {
    const view = { x: 20, y: 40, scale: 0.5 };
    const anchor = { x: 300, y: 200 };
    const zoomed = zoomViewport(view, 1.2, anchor);
    expect((anchor.x - zoomed.x) / zoomed.scale).toBeCloseTo((anchor.x - view.x) / view.scale);
    expect((anchor.y - zoomed.y) / zoomed.scale).toBeCloseTo((anchor.y - view.y) / view.scale);
    expect(zoomViewport(view, 100, anchor).scale).toBe(MAX_SCALE);
    expect(zoomViewport(view, 0, anchor).scale).toBe(MIN_SCALE);
  });
});
