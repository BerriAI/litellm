import { render } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it } from "vitest";
import { ChartContainer } from "./chart";

describe("ChartStyle hardening", () => {
  it("sanitizes config keys and strips structural characters from color values", () => {
    const { container } = render(
      <ChartContainer
        config={{
          "metrics.total_tokens": { label: "Total Tokens", color: "var(--color-blue-500, #3b82f6)" },
          "evil}key": { label: "evil", color: "red;}</style><img src=x onerror=alert(1)>" },
        }}
      >
        <svg />
      </ChartContainer>,
    );

    const style = container.querySelector("style");
    expect(style).not.toBeNull();
    const css = style!.innerHTML;

    expect(css).toContain("--color-metrics_total_tokens: var(--color-blue-500, #3b82f6);");
    expect(css).not.toContain("metrics.total_tokens");
    expect(css).toContain("--color-evil_key:");
    expect(css).not.toContain("<");
    expect((css.match(/{/g) ?? []).length).toBe((css.match(/}/g) ?? []).length);
  });

  it("emits no style tag when no config entry has a color", () => {
    const { container } = render(
      <ChartContainer config={{ passed: { label: "passed" } }}>
        <svg />
      </ChartContainer>,
    );
    expect(container.querySelector("style")).toBeNull();
  });
});
