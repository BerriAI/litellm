import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TerminologyProvider, useTerminology } from "./TerminologyContext";

const uiConfig = vi.hoisted(() => ({ data: undefined as { use_si_terminology?: boolean } | undefined }));

vi.mock("@/app/(dashboard)/hooks/uiConfig/useUIConfig", () => ({
  useUIConfig: () => ({ data: uiConfig.data }),
}));

function GatewayTitle() {
  const term = useTerminology();
  return <h1>{term("AI Gateway")}</h1>;
}

describe("TerminologyProvider", () => {
  it.each([
    [{ use_si_terminology: true }, "SI Gateway"],
    [{ use_si_terminology: false }, "AI Gateway"],
    [{}, "AI Gateway"],
    [undefined, "AI Gateway"],
  ])("with UI config %j renders %j", (data, expected) => {
    uiConfig.data = data;
    render(
      <TerminologyProvider>
        <GatewayTitle />
      </TerminologyProvider>,
    );
    expect(screen.getByRole("heading")).toHaveTextContent(expected);
  });

  it("leaves copy unchanged for components rendered outside the provider", () => {
    uiConfig.data = { use_si_terminology: true };
    render(<GatewayTitle />);
    expect(screen.getByRole("heading")).toHaveTextContent("AI Gateway");
  });
});
