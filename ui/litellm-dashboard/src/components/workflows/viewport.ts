export interface Size {
  width: number;
  height: number;
}

export interface Viewport {
  x: number;
  y: number;
  scale: number;
}

export const NODE_WIDTH = 288;
export const NODE_HEIGHT = 112;
export const NODE_GAP = 64;
export const GRAPH_WIDTH = 400;
export const MIN_SCALE = 0.1;
export const MAX_SCALE = 2;

export function nodePosition(index: number) {
  return { x: index % 2 === 0 ? 44 : 68, y: 100 + index * (NODE_HEIGHT + NODE_GAP) };
}

export function graphSize(nodeCount: number): Size {
  return { width: GRAPH_WIDTH, height: 100 + nodeCount * (NODE_HEIGHT + NODE_GAP) + 24 };
}

export function fitViewport(container: Size, graph: Size): Viewport {
  const scale = Math.min(
    1,
    Math.max(1, container.width - 64) / graph.width,
    Math.max(1, container.height - 128) / graph.height,
  );
  return { x: (container.width - graph.width * scale) / 2, y: (container.height - graph.height * scale) / 2, scale };
}

export function zoomViewport(
  view: Viewport,
  factor: number,
  anchor: { x: number; y: number },
  minimum = MIN_SCALE,
): Viewport {
  const scale = Math.max(Math.min(minimum, view.scale), Math.min(MAX_SCALE, view.scale * factor));
  const ratio = scale / view.scale;
  return { x: anchor.x - (anchor.x - view.x) * ratio, y: anchor.y - (anchor.y - view.y) * ratio, scale };
}
