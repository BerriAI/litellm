import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectFlowBanner from "./ConnectFlowBanner";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "https://gateway.example.com",
}));

afterEach(() => {
  vi.restoreAllMocks();
  sessionStorage.clear();
});

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

  it("does NOT complete the flow on pagehide (completion requires the explicit button)", () => {
    // Security regression: an attacker could lure a signed-in victim to their own client's
    // authorize URL; the victim merely closing the tab must NOT deliver a victim-bound code.
    // Completion is a deliberate button press, never a side effect of leaving the page.
    const beaconMock = vi.fn(() => true);
    vi.stubGlobal("navigator", { ...navigator, sendBeacon: beaconMock });
    render(<ConnectFlowBanner flowHandle="flow-xyz" clientOrigin="https://claude.ai" />);

    window.dispatchEvent(new Event("pagehide"));

    expect(beaconMock).not.toHaveBeenCalled();
  });
});
