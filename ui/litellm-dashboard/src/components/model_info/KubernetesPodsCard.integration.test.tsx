import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, testQueryClient } from "@/../tests/test-utils";
import KubernetesPodsCard from "./KubernetesPodsCard";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";

function PodLanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

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

  it("切换语言更新 Pod 说明但保留服务主机和 IP", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="kubernetes-language" storageScope="model-info-language">
        <PodLanguageSwitch />
        <KubernetesPodsCard modelId="model-pods-1" />
      </LanguageProvider>,
    );
    expect(await screen.findByText("3 个就绪 Pod")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "就绪 Kubernetes Pod 的 IP 地址" })).toBeInTheDocument();
    expect(screen.getByText("vllm-headless.ns.svc.cluster.local:8000")).toBeInTheDocument();
    expect(screen.getByText("10.0.0.1")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText("3 ready pods")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(screen.getByText("10.0.0.1")).toBeInTheDocument();
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
    expect(screen.queryByText("0 ready pods")).not.toBeInTheDocument();
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
