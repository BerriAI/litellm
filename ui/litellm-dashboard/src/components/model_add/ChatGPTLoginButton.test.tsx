import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ChatGPTLoginButton from "./ChatGPTLoginButton";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/components/networking", async () => {
  const actual = await vi.importActual("@/components/networking");
  return {
    ...actual,
    chatgptOauthStartCall: vi.fn(),
    chatgptOauthStatusCall: vi.fn(),
    chatgptOauthCancelCall: vi.fn(),
  };
});

describe("ChatGPTLoginButton", () => {
  it("renders with the ChatGPT provider label", () => {
    render(<ChatGPTLoginButton credentialName="c" onSuccess={vi.fn()} />);
    expect(
      screen.getByRole("button", { name: /Sign in with ChatGPT/i }),
    ).toBeInTheDocument();
  });
});
