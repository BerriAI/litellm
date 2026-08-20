import { act, fireEvent, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import AdvancedSettings from "./advanced_settings";

const mockUsePtuCostAttributionEnabled = vi.fn();

vi.mock("@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled", () => ({
  usePtuCostAttributionEnabled: () => mockUsePtuCostAttributionEnabled(),
}));

const PTU_LABELS = ["PTU Count", "Calculated Cost per PTU / Hour (USD)", "PTU Effective From (UTC)"];

const renderAdvancedSettings = () =>
  render(
    <MountedFormHost>
      <AdvancedSettings
        showAdvancedSettings={true}
        setShowAdvancedSettings={() => {}}
        guardrailsList={[]}
        tagsList={{}}
        accessToken="test-token"
      />
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

  it("should render tags list", async () => {
    const { getByText } = renderAdvancedSettings();
    fireEvent.click(getByText("Advanced Settings"));
    await waitFor(() => {
      expect(getByText("Tags")).toBeInTheDocument();
    });
  });

  it("should render the litellm params", async () => {
    const { getByText } = renderAdvancedSettings();
    act(() => {
      fireEvent.click(getByText("Advanced Settings"));
    });
    await waitFor(() => {
      expect(getByText("LiteLLM Params")).toBeInTheDocument();
    });
  });

  it("hides every PTU field when PTU cost attribution is disabled", async () => {
    const { getByText, queryByText } = renderAdvancedSettings();
    act(() => {
      fireEvent.click(getByText("Advanced Settings"));
    });
    await waitFor(() => {
      expect(getByText("Tags")).toBeInTheDocument();
    });

    for (const label of PTU_LABELS) {
      expect(queryByText(label)).not.toBeInTheDocument();
    }
    expect(queryByText("PTU Effective To (UTC)")).not.toBeInTheDocument();
  });

  it("shows every PTU field when PTU cost attribution is enabled", async () => {
    mockUsePtuCostAttributionEnabled.mockReturnValue(true);
    const { getByText } = renderAdvancedSettings();
    act(() => {
      fireEvent.click(getByText("Advanced Settings"));
    });

    await waitFor(() => {
      expect(getByText("PTU Count")).toBeInTheDocument();
    });
    for (const label of PTU_LABELS) {
      expect(getByText(label)).toBeInTheDocument();
    }
    expect(getByText("PTU Effective To (UTC)")).toBeInTheDocument();
  });
});
