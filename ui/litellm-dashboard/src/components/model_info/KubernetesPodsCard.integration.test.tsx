import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, testQueryClient } from "@/../tests/test-utils";
import KubernetesPodsCard from "./KubernetesPodsCard";

const podResponse = {
  model_id: "model-pods-1",
  service_host: "vllm-headless.ns.svc.cluster.local",
  port: 8000,
  pod_ips: ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
  pod_count: 3,
  error: null,
};

const lookupErrorResponse = {
  ...podResponse,
  pod_ips: [],
  pod_count: 0,
  error: "temporary DNS failure",
};

const network = vi.fn<typeof fetch>();

describe("KubernetesPodsCard", () => {
  beforeEach(() => {
    testQueryClient.clear();
    network.mockReset().mockResolvedValue(Response.json(podResponse));
    vi.stubGlobal("fetch", network);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    testQueryClient.clear();
  });

  it("shows the service host, ready count, and all pod IPs", async () => {
    renderWithProviders(<KubernetesPodsCard modelId="model-pods-1" />);

    expect(await screen.findByText("vllm-headless.ns.svc.cluster.local:8000")).toBeInTheDocument();
    expect(screen.getByText("3 ready pods")).toBeInTheDocument();
    expect(screen.getByText("10.0.0.1")).toBeInTheDocument();
    expect(screen.getByText("10.0.0.2")).toBeInTheDocument();
    expect(screen.getByText("10.0.0.3")).toBeInTheDocument();
  });

  it("shows an empty state when no pods are ready", async () => {
    network.mockResolvedValueOnce(
      Response.json({
        ...podResponse,
        pod_ips: [],
        pod_count: 0,
      }),
    );

    renderWithProviders(<KubernetesPodsCard modelId="model-pods-1" />);

    expect(await screen.findByText("No ready pods")).toBeInTheDocument();
  });

  it("shows a pod lookup error", async () => {
    network.mockResolvedValueOnce(Response.json(lookupErrorResponse));

    renderWithProviders(<KubernetesPodsCard modelId="model-pods-1" />);

    expect(await screen.findByRole("alert")).toHaveTextContent("temporary DNS failure");
    expect(screen.queryByText("0 ready pods")).not.toBeInTheDocument();
  });

  it("shows an API error", async () => {
    network.mockResolvedValueOnce(Response.json({ detail: "DNS lookup unavailable" }, { status: 503 }));

    renderWithProviders(<KubernetesPodsCard modelId="model-pods-1" />);

    expect(await screen.findByRole("alert")).toHaveTextContent("DNS lookup unavailable");
  });

  it("shows the API message for an ineligible Kubernetes service host", async () => {
    const message =
      "Kubernetes pod discovery is enabled for this model, but its api_base is not an http:// " +
      "service hostname, so no pods are resolved";
    network.mockResolvedValueOnce(
      Response.json(
        {
          error: {
            message,
            type: "bad_request_error",
            param: "model_id",
            code: "400",
          },
        },
        { status: 400 },
      ),
    );

    renderWithProviders(<KubernetesPodsCard modelId="model-pods-1" />);

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
  });

  it("refreshes the pod lookup when requested", async () => {
    const user: ReturnType<typeof userEvent.setup> = userEvent.setup();
    renderWithProviders(<KubernetesPodsCard modelId="model-pods-1" />);

    expect(await screen.findByText("10.0.0.1")).toBeInTheDocument();
    network.mockResolvedValueOnce(
      Response.json({
        ...podResponse,
        pod_ips: ["10.0.0.4"],
        pod_count: 1,
      }),
    );
    await user.click(screen.getByRole("button", { name: "Refresh" }));

    expect(await screen.findByText("10.0.0.4")).toBeInTheDocument();
    expect(network).toHaveBeenCalledTimes(2);
  });
});
