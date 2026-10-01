import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { OtelEndpointPanel, proxyHost } from "./OtelEndpointPanel";
import { tracingEnvSnippet } from "./TracingSetupCard";

vi.mock("../../networking", () => ({ getProxyBaseUrl: () => "http://127.0.0.1:4012/" }));
vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

const openPanel = async (newestRunStart: string | null, onOpenGuide = vi.fn()) => {
  const user = userEvent.setup();
  render(<OtelEndpointPanel newestRunStart={newestRunStart} onOpenGuide={onOpenGuide} />);
  await user.click(screen.getByRole("button", { name: "OpenTelemetry endpoint" }));
  return { user, onOpenGuide };
};

describe("OtelEndpointPanel", () => {
  it("shows the proxy host in the pill without opening it", () => {
    render(<OtelEndpointPanel newestRunStart={null} onOpenGuide={vi.fn()} />);
    const pill = screen.getByRole("button", { name: "OpenTelemetry endpoint" });
    expect(pill).toHaveTextContent("OTLP");
    expect(pill).toHaveTextContent("127.0.0.1:4012");
  });

  it("opens the endpoint, the ingest route, the auth scope and the env vars", async () => {
    await openPanel(null);
    expect(await screen.findByText("OpenTelemetry endpoint")).toBeInTheDocument();
    expect(screen.getByText("http://127.0.0.1:4012")).toBeInTheDocument();
    expect(screen.getByText("POST http://127.0.0.1:4012/v1/traces")).toBeInTheDocument();
    expect(screen.getByText(/Runs are scoped to that key and its team/)).toBeInTheDocument();
    const env = screen.getByText(/OTEL_EXPORTER_OTLP_HEADERS/);
    expect(env).toHaveTextContent("export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4012 export");
    expect(env).toHaveTextContent("Authorization=Bearer $LITELLM_API_KEY");
  });

  it("reports whether the loaded range has runs", async () => {
    await openPanel(null);
    expect(await screen.findByTestId("otel-status")).toHaveTextContent("No runs in this range");
  });

  it("reports the newest run when traces are arriving", async () => {
    await openPanel(new Date(Date.now() - 3 * 60_000).toISOString());
    expect(await screen.findByTestId("otel-status")).toHaveTextContent("Receiving traces, newest run 3m ago");
  });

  it("opens the full setup guide", async () => {
    const { user, onOpenGuide } = await openPanel(null);
    await user.click(await screen.findByRole("button", { name: /Full setup guide/ }));
    expect(onOpenGuide).toHaveBeenCalledTimes(1);
  });
});

describe("pill env snippet", () => {
  it("is the setup guide's snippet: base URL endpoint, protocol, key placeholder, no /v1/traces suffix", () => {
    const env = tracingEnvSnippet("http://127.0.0.1:4012");
    expect(env).toContain("OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf");
    expect(env).toContain("OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4012\n");
    expect(env).not.toContain("/v1/traces");
    expect(env).toContain("Authorization=Bearer $LITELLM_API_KEY");
    expect(env).not.toMatch(/sk-[A-Za-z0-9]/);
  });
});

describe("proxyHost", () => {
  it("shows host and port, and falls back to the raw value when it is not a URL", () => {
    expect(proxyHost("https://gateway.example.com")).toBe("gateway.example.com");
    expect(proxyHost("http://127.0.0.1:4012")).toBe("127.0.0.1:4012");
    expect(proxyHost("")).toBe("");
  });
});
