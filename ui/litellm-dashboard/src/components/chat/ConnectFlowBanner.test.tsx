import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectFlowBanner, { isLoopbackOrigin } from "./ConnectFlowBanner";

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
    expect(form).toHaveAttribute("method", "POST");
    expect(form).toHaveAttribute("action", "https://gateway.example.com/authorize/complete");

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

  it("offers manual delivery for a loopback client, posted only when checked", () => {
    const { container } = render(
      <ConnectFlowBanner flowHandle="flow-handle-123" clientOrigin="http://localhost:3118" />,
    );

    const checkbox = container.querySelector('input[type="checkbox"][name="delivery"]') as HTMLInputElement;
    expect(checkbox).not.toBeNull();
    expect(checkbox.value).toBe("manual");
    expect(checkbox.checked).toBe(false);
    expect(screen.getByText(/remote or SSH machine/i)).toBeInTheDocument();
  });

  it("does not offer manual delivery for a routable client origin or an unknown one", () => {
    const routable = render(<ConnectFlowBanner flowHandle="h" clientOrigin="https://claude.ai" />);
    expect(routable.container.querySelector('input[name="delivery"]')).toBeNull();

    const unknown = render(<ConnectFlowBanner flowHandle="h" clientOrigin={null} />);
    expect(unknown.container.querySelector('input[name="delivery"]')).toBeNull();
  });

  it("classifies loopback origins like the server does", () => {
    expect(isLoopbackOrigin("http://localhost:3118")).toBe(true);
    expect(isLoopbackOrigin("http://127.0.0.1:8080")).toBe(true);
    expect(isLoopbackOrigin("http://127.5.4.3:1")).toBe(true);
    expect(isLoopbackOrigin("http://[::1]:9000")).toBe(true);
    expect(isLoopbackOrigin("http://[0:0:0:0:0:0:0:1]:9000")).toBe(true);
    expect(isLoopbackOrigin("https://claude.ai")).toBe(false);
    expect(isLoopbackOrigin("http://127.evil.com")).toBe(false);
    expect(isLoopbackOrigin("http://localhost.evil.com")).toBe(false);
    expect(isLoopbackOrigin(null)).toBe(false);
    expect(isLoopbackOrigin("not a url")).toBe(false);
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
