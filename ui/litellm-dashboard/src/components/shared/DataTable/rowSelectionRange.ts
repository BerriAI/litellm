import type { RowSelectionState } from "@tanstack/react-table";

export const idsBetween = (orderedIds: readonly string[], fromId: string, toId: string): readonly string[] => {
  const from = orderedIds.indexOf(fromId);
  const to = orderedIds.indexOf(toId);
  if (from === -1 || to === -1) return [];
  return orderedIds.slice(Math.min(from, to), Math.max(from, to) + 1);
};

export const paintRowSelection = (
  snapshot: RowSelectionState,
  rangeIds: readonly string[],
  selected: boolean,
): RowSelectionState => {
  const range: ReadonlySet<string> = new Set(rangeIds);
  const outsideRange = Object.entries(snapshot).filter(([id]) => !range.has(id));
  const painted = selected ? rangeIds.map((id): [string, boolean] => [id, true]) : [];
  return Object.fromEntries([...outsideRange, ...painted]);
};
