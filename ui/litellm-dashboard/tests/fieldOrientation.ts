import { expect } from "vitest";

export const ROW_LAYOUT_CLASSES = ["flex-row", "items-center"] as const;
export const STRETCH_CHILDREN_CLASS = "*:w-full";

/**
 * Asserts a control sits beside its label at its own width instead of being stretched across the
 * field. Reaches for the resolved classes because the defect is purely visual: nothing accessible
 * distinguishes a square checkbox from a full-width bar.
 */
export const expectControlBesideLabel = (control: HTMLElement): void => {
  const field = control.closest('[data-slot="field"]');
  if (field === null) throw new Error("control is not rendered inside a form field");

  expect(field).toHaveAttribute("data-orientation", "horizontal");
  expect(field).toHaveClass(...ROW_LAYOUT_CLASSES);
  expect(field).not.toHaveClass(STRETCH_CHILDREN_CLASS);
};
