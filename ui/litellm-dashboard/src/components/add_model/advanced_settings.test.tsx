import { fireEvent, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AdvancedSettings from "./advanced_settings";
import { Tag } from "../tag_management/types";

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
});
