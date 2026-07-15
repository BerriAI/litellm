import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ConnectFlowBanner from "./ConnectFlowBanner";
import { PERSERVER_CONNECTING_KEY } from "@/hooks/mcpOAuthUtils";

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

  it("best-effort auto-finishes on pagehide (closing the tab)", () => {
    const beaconMock = vi.fn(() => true);
    vi.stubGlobal("navigator", { ...navigator, sendBeacon: beaconMock });
    render(<ConnectFlowBanner flowHandle="flow-xyz" clientOrigin="https://claude.ai" />);

    window.dispatchEvent(new Event("pagehide"));

    expect(beaconMock).toHaveBeenCalledTimes(1);
    const [url, body] = beaconMock.mock.calls[0] as unknown as [string, URLSearchParams];
    expect(url).toBe("https://gateway.example.com/authorize/complete");
    expect(body.toString()).toContain("flow=flow-xyz");
  });

  it("does NOT auto-finish while a per-server connect is navigating away", () => {
    const beaconMock = vi.fn(() => true);
    vi.stubGlobal("navigator", { ...navigator, sendBeacon: beaconMock });
    render(<ConnectFlowBanner flowHandle="flow-xyz" clientOrigin="https://claude.ai" />);

    // the per-server connect flow sets this right before it navigates to the upstream IdP
    sessionStorage.setItem(PERSERVER_CONNECTING_KEY, "1");
    window.dispatchEvent(new Event("pagehide"));

    expect(beaconMock).not.toHaveBeenCalled();
  });

  it("does NOT double-fire the auto-finish after the button was pressed", () => {
    const beaconMock = vi.fn(() => true);
    vi.stubGlobal("navigator", { ...navigator, sendBeacon: beaconMock });
    const { container } = render(<ConnectFlowBanner flowHandle="flow-xyz" clientOrigin="https://claude.ai" />);

    // jsdom does not submit forms; fire the form's submit so onSubmit marks it finished
    fireEvent.submit(container.querySelector("form")!);
    window.dispatchEvent(new Event("pagehide"));

    expect(beaconMock).not.toHaveBeenCalled();
  });
});
