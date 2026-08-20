import { act, renderHook } from "@testing-library/react";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { useSyntaxTheme, type SyntaxTheme } from "./useSyntaxTheme";

const callerLightTheme: SyntaxTheme = { 'code[class*="language-"]': { color: "rebeccapurple" } };

const setRootDark = async (enabled: boolean) => {
  await act(async () => {
    document.documentElement.classList.toggle("dark", enabled);
    await Promise.resolve();
  });
};

beforeEach(() => {
  document.documentElement.classList.remove("dark");
});

afterAll(() => {
  document.documentElement.classList.remove("dark");
});

describe("useSyntaxTheme", () => {
  it("keeps the caller's own stylesheet in light mode", () => {
    const { result } = renderHook(() => useSyntaxTheme(callerLightTheme));

    expect(result.current).toBe(callerLightTheme);
  });

  it("swaps to oneDark when the root element turns dark", async () => {
    const { result } = renderHook(() => useSyntaxTheme(callerLightTheme));

    await setRootDark(true);

    expect(result.current).toBe(oneDark);
  });

  it("restores the caller's stylesheet when dark mode is turned back off", async () => {
    document.documentElement.classList.add("dark");
    const { result } = renderHook(() => useSyntaxTheme(callerLightTheme));
    expect(result.current).toBe(oneDark);

    await setRootDark(false);

    expect(result.current).toBe(callerLightTheme);
  });
});
