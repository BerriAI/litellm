import { act, fireEvent, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AdvancedSettings from "./advanced_settings";

describe("AdvancedSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  it("should render", () => {
    render(
      <AdvancedSettings
        showAdvancedSettings={true}
        setShowAdvancedSettings={() => {}}
        guardrailsList={[]}
        tagsList={{}}
      />,
    );
  });

  it("should render tags list", async () => {
    const { getByText } = render(
      <AdvancedSettings
        showAdvancedSettings={true}
        setShowAdvancedSettings={() => {}}
        guardrailsList={[]}
        tagsList={{}}
      />,
    );
    fireEvent.click(getByText("Advanced Settings"));
    await waitFor(() => {
      expect(getByText("Tags")).toBeInTheDocument();
    });
  });

  it("should render the litellm params", async () => {
    const { getByText } = render(
      <AdvancedSettings
        showAdvancedSettings={true}
        setShowAdvancedSettings={() => {}}
        guardrailsList={[]}
        tagsList={{}}
      />,
    );
    act(() => {
      fireEvent.click(getByText("Advanced Settings"));
    });
    await waitFor(() => {
      expect(getByText("LiteLLM Params")).toBeInTheDocument();
    });
  });
});
