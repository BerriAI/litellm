import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import SignedIdentityHint from "./SignedIdentityHint";
import { AUTH_TYPE } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({ getProxyBaseUrl: () => "https://litellm.example.com" }));

describe("SignedIdentityHint", () => {
  it("tells the server owner where to verify the token and which audience to expect", () => {
    render(<SignedIdentityHint authType={AUTH_TYPE.SIGNED_IDENTITY} serverUrl="https://reports.example.com/mcp" />);

    expect(screen.getByText("https://litellm.example.com/.well-known/jwks.json")).toBeInTheDocument();
    expect(screen.getByText("https://reports.example.com/mcp")).toBeInTheDocument();
    expect(screen.getByText(/caller's email/)).toBeInTheDocument();
  });

  it("asks for the server URL when none is entered yet", () => {
    render(<SignedIdentityHint authType={AUTH_TYPE.SIGNED_IDENTITY} serverUrl="" />);

    expect(screen.getByText("This server's URL, once entered above")).toBeInTheDocument();
  });

  it("renders nothing for any other auth type", () => {
    const { container } = render(<SignedIdentityHint authType={AUTH_TYPE.BEARER_TOKEN} serverUrl="https://x/mcp" />);
    expect(container).toBeEmptyDOMElement();
  });
});
