import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import TruePassthroughWarning from "./TruePassthroughWarning";
import { AUTH_TYPE } from "@/components/mcp_tools/types";

describe("TruePassthroughWarning", () => {
  it("warns when auth type is true_passthrough", () => {
    render(<TruePassthroughWarning authType={AUTH_TYPE.TRUE_PASSTHROUGH} />);

    expect(screen.getByText("True Passthrough disables LiteLLM authentication for this server")).toBeInTheDocument();
    expect(screen.getByText(/Anyone who can reach the gateway can call this server/)).toBeInTheDocument();
  });

  it("renders nothing for any other auth type", () => {
    const { container } = render(<TruePassthroughWarning authType={AUTH_TYPE.OAUTH2} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when no auth type is set", () => {
    const { container } = render(<TruePassthroughWarning authType={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
