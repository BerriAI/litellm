import { act, renderHook } from "@testing-library/react";
import { ThemeProvider, useTheme } from "next-themes";
import type { ReactNode } from "react";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { useSyntaxTheme, type SyntaxTheme } from "./useSyntaxTheme";

const callerLightTheme: SyntaxTheme = { 'code[class*="language-"]': { color: "rebeccapurple" } };

const renderSyntaxTheme = (defaultTheme: string) =>
  renderHook(() => ({ syntax: useSyntaxTheme(callerLightTheme), setTheme: useTheme().setTheme }), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <ThemeProvider attribute="class" enableSystem={false} defaultTheme={defaultTheme}>
        {children}
      </ThemeProvider>
    ),
  });

beforeEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove("dark", "light");
});

afterAll(() => {
  document.documentElement.classList.remove("dark", "light");
});

describe("useSyntaxTheme", () => {
  it("keeps the caller's own stylesheet in light mode", () => {
    const { result } = renderSyntaxTheme("light");

    expect(result.current.syntax).toBe(callerLightTheme);
  });

  it("serves oneDark when the resolved theme is dark", () => {
    const { result } = renderSyntaxTheme("dark");

    expect(result.current.syntax).toBe(oneDark);
  });

  it("swaps stylesheets when the theme is changed at runtime", () => {
    const { result } = renderSyntaxTheme("light");

    act(() => result.current.setTheme("dark"));
    expect(result.current.syntax).toBe(oneDark);

    act(() => result.current.setTheme("light"));
    expect(result.current.syntax).toBe(callerLightTheme);
  });
});
