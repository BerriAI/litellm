import { renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { stubViewportMatchMedia } from "../../../tests/viewport";
import { useIsNarrow } from "./useIsNarrow";

afterEach(() => vi.unstubAllGlobals());

it("keeps the catalog narrow through its boundary and switches on resize", async () => {
  const resizeTo = stubViewportMatchMedia();
  await resizeTo(1280);
  const { result } = renderHook(useIsNarrow);
  expect(result.current).toBe(false);
  await resizeTo(1100);
  expect(result.current).toBe(true);
  await resizeTo(1100.5);
  expect(result.current).toBe(false);
  await resizeTo(375);
  expect(result.current).toBe(true);
});
