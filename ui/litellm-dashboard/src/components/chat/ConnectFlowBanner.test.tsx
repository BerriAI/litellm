import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectFlowBanner from "./ConnectFlowBanner";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "https://gateway.example.com",
}));

describe("ConnectFlowBanner", () => {
  it("posts the flow handle to the proxy /authorize/complete as a full-page form", () => {
    const { container } = render(<ConnectFlowBanner flowHandle="flow-handle-123" clientOrigin="https://claude.ai" />);

    const form = container.querySelector("form")!;
    expect(form.getAttribute("method")).toBe("POST");
    expect(form.getAttribute("action")).toBe("https://gateway.example.com/authorize/complete");

    const hidden = form.querySelector('input[name="flow"]') as HTMLInputElement;
    expect(hidden.value).toBe("flow-handle-123");
    // No token, code, or secret is ever placed in the form; the sealed cookie carries them.
    expect(form.innerHTML).not.toContain("token");
  });

  it("shows the client origin so the user knows what they are connecting to", () => {
    render(<ConnectFlowBanner flowHandle="h" clientOrigin="https://claude.ai" />);
    expect(screen.getAllByText(/claude\.ai/).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /finish connecting/i })).toBeInTheDocument();
  });

  it("falls back to a generic label when the client origin is unknown", () => {
    render(<ConnectFlowBanner flowHandle="h" clientOrigin={null} />);
    expect(screen.getAllByText(/the application/).length).toBeGreaterThan(0);
  });
});
