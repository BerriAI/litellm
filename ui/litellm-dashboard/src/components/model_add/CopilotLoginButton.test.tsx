import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import CopilotLoginButton from "./CopilotLoginButton";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/components/networking", async () => {
  const actual = await vi.importActual("@/components/networking");
  return {
    ...actual,
    copilotOauthStartCall: vi.fn(),
    copilotOauthStatusCall: vi.fn(),
    copilotOauthCancelCall: vi.fn(),
  };
});

describe("CopilotLoginButton", () => {
  it("renders with the GitHub Copilot provider label", () => {
    render(<CopilotLoginButton credentialName="c" onSuccess={vi.fn()} />);
    expect(
      screen.getByRole("button", { name: /Sign in with GitHub Copilot/i }),
    ).toBeInTheDocument();
  });
});
