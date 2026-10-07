import { act, fireEvent, render, waitFor, screen } from "@testing-library/react";
import { useFormContext } from "react-hook-form";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import type { MountedFormValues } from "../common_components/MountedFormField";
import AdvancedSettings from "./advanced_settings";

const mockUsePtuCostAttributionEnabled = vi.fn();

vi.mock("@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled", () => ({
  usePtuCostAttributionEnabled: () => mockUsePtuCostAttributionEnabled(),
}));

const PTU_LABELS = ["PTU Count", "Calculated Cost per PTU / Hour (USD)", "PTU Effective From (UTC)"];

const KubernetesPodDiscoveryValue = () => {
  const form = useFormContext<MountedFormValues>();
  return (
    <output aria-label="Kubernetes pod discovery form value" role="status">
      {String(form.watch("kubernetes_pod_discovery") === true)}
    </output>
  );
};

const renderAdvancedSettings = (includeDiscoveryValue = false) =>
  render(
    <MountedFormHost>
      <AdvancedSettings
        showAdvancedSettings={true}
        setShowAdvancedSettings={() => {}}
        guardrailsList={[]}
        tagsList={{}}
        accessToken="test-token"
      />
      {includeDiscoveryValue && <KubernetesPodDiscoveryValue />}
    </MountedFormHost>,
  );

describe("AdvancedSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUsePtuCostAttributionEnabled.mockReturnValue(false);
  });

  it("should render", () => {
    renderAdvancedSettings();
  });

  it("updates the Kubernetes pod discovery form value when enabled", async () => {
    renderAdvancedSettings(true);
    act(() => {
      fireEvent.click(screen.getByText("Advanced Settings"));
    });

    const toggle = await screen.findByRole("switch", { name: "Kubernetes pod discovery" });
    const formValue = screen.getByRole("status", { name: "Kubernetes pod discovery form value" });
    expect(toggle).not.toBeChecked();
    expect(formValue).toHaveTextContent("false");

    fireEvent.click(toggle);

    expect(toggle).toBeChecked();
    expect(formValue).toHaveTextContent("true");
  });

  it("should render tags list", async () => {
    renderAdvancedSettings();
    fireEvent.click(screen.getByText("Advanced Settings"));
    await waitFor(() => {
      expect(screen.getByText("Tags")).toBeInTheDocument();
    });
  });

  it("should render the litellm params", async () => {
    renderAdvancedSettings();
    act(() => {
      fireEvent.click(screen.getByText("Advanced Settings"));
    });
    await waitFor(() => {
      expect(screen.getByText("LiteLLM Params")).toBeInTheDocument();
    });
  });

  it("hides every PTU field when PTU cost attribution is disabled", async () => {
    renderAdvancedSettings();
    act(() => {
      fireEvent.click(screen.getByText("Advanced Settings"));
    });
    await waitFor(() => {
      expect(screen.getByText("Tags")).toBeInTheDocument();
    });

    for (const label of PTU_LABELS) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
    expect(screen.queryByText("PTU Effective To (UTC)")).not.toBeInTheDocument();
  });

  it("shows every PTU field when PTU cost attribution is enabled", async () => {
    mockUsePtuCostAttributionEnabled.mockReturnValue(true);
    renderAdvancedSettings();
    act(() => {
      fireEvent.click(screen.getByText("Advanced Settings"));
    });

    await waitFor(() => {
      expect(screen.getByText("PTU Count")).toBeInTheDocument();
    });
    for (const label of PTU_LABELS) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.getByText("PTU Effective To (UTC)")).toBeInTheDocument();
  });
});
