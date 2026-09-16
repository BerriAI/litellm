import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useWatch } from "react-hook-form";
import type { MountedFormValues } from "../common_components/MountedFormField";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import SapDeploymentDiscovery from "./SapDeploymentDiscovery";
import { listSapDeploymentsCall } from "../networking";

vi.mock("../networking", () => ({
  listSapDeploymentsCall: vi.fn(),
}));

const listSapDeploymentsMock = vi.mocked(listSapDeploymentsCall);

const _MODEL = "anthropic--claude-4.8-opus";
const _DEPLOYMENT_URL = "https://api.example.com/v2/inference/deployments/d38af17dc133a768";

const _DEPLOYMENT = {
  model_name: _MODEL,
  deployment_url: _DEPLOYMENT_URL,
  id: "d38af17dc133a768",
  status: "RUNNING",
  created_at: "2026-07-14T02:44:50Z",
};

const SelectionProbe = () => {
  const model = useWatch<MountedFormValues>({ name: "model" });
  const apiBase = useWatch<MountedFormValues>({ name: "api_base" });
  return (
    <>
      <output data-testid="model">{JSON.stringify(model ?? null)}</output>
      <output data-testid="api-base">{String(apiBase ?? "")}</output>
    </>
  );
};

const renderDiscovery = (defaultValues: MountedFormValues) =>
  render(
    <MountedFormHost defaultValues={defaultValues}>
      <SapDeploymentDiscovery accessToken="admin-token" />
      <SelectionProbe />
    </MountedFormHost>,
  );

beforeEach(() => {
  listSapDeploymentsMock.mockReset();
});

describe("SapDeploymentDiscovery", () => {
  it("lists the deployments the service key can see", async () => {
    listSapDeploymentsMock.mockResolvedValue([_DEPLOYMENT]);
    renderDiscovery({ api_key: "svc-key" });

    fireEvent.click(screen.getByRole("button", { name: "Discover deployments" }));

    expect(await screen.findByText(_MODEL)).toBeInTheDocument();
    expect(listSapDeploymentsMock).toHaveBeenCalledWith("admin-token", "svc-key", undefined);
  });

  it("forwards the entered resource group into the discovery call", async () => {
    listSapDeploymentsMock.mockResolvedValue([_DEPLOYMENT]);
    renderDiscovery({ api_key: "svc-key" });

    fireEvent.change(screen.getByLabelText("AI Resource Group"), { target: { value: "team-a" } });
    fireEvent.click(screen.getByRole("button", { name: "Discover deployments" }));

    expect(await screen.findByText(_MODEL)).toBeInTheDocument();
    expect(listSapDeploymentsMock).toHaveBeenCalledWith("admin-token", "svc-key", "team-a");
  });

  it("pins the model and api_base when a deployment is selected", async () => {
    listSapDeploymentsMock.mockResolvedValue([_DEPLOYMENT]);
    renderDiscovery({ api_key: "svc-key" });

    fireEvent.click(screen.getByRole("button", { name: "Discover deployments" }));
    fireEvent.click(await screen.findByText(_MODEL));

    await waitFor(() => expect(screen.getByTestId("api-base")).toHaveTextContent(_DEPLOYMENT_URL));
    expect(screen.getByTestId("model")).toHaveTextContent(JSON.stringify([`sap/deployment/${_MODEL}`]));
  });

  it("marks the chosen deployment as selected and confirms the fill", async () => {
    const _OTHER = {
      model_name: "gpt-5.6-sol",
      deployment_url: "https://api.example.com/v2/inference/deployments/aaaa1111bbbb2222",
      id: "aaaa1111bbbb2222",
      status: "RUNNING",
      created_at: "2026-07-14T02:44:50Z",
    };
    listSapDeploymentsMock.mockResolvedValue([_DEPLOYMENT, _OTHER]);
    renderDiscovery({ api_key: "svc-key" });

    fireEvent.click(screen.getByRole("button", { name: "Discover deployments" }));
    fireEvent.click(await screen.findByText(_MODEL));

    const deploymentButtons = screen
      .getAllByRole("button")
      .filter((b) => b.getAttribute("aria-pressed") !== null);
    const chosen = deploymentButtons.find((b) => b.textContent?.includes(_MODEL));
    const other = deploymentButtons.find((b) => b.textContent?.includes(_OTHER.model_name));
    await waitFor(() => expect(chosen).toHaveAttribute("aria-pressed", "true"));
    expect(other).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText(/Model fields above are filled in/i)).toBeInTheDocument();
  });

  it("refuses to call the proxy when no service key is entered", async () => {
    renderDiscovery({});

    fireEvent.click(screen.getByRole("button", { name: "Discover deployments" }));

    expect(await screen.findByText(/service key above first/i)).toBeInTheDocument();
    expect(listSapDeploymentsMock).not.toHaveBeenCalled();
  });

  it("surfaces the failure when discovery is rejected", async () => {
    listSapDeploymentsMock.mockRejectedValue(new Error("Failed to obtain an SAP AI Core token from the service key."));
    renderDiscovery({ api_key: "svc-key" });

    fireEvent.click(screen.getByRole("button", { name: "Discover deployments" }));

    expect(await screen.findByText(/Failed to obtain an SAP AI Core token/i)).toBeInTheDocument();
  });
});
