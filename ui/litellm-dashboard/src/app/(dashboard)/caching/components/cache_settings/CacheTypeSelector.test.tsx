import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import CacheTypeSelector from "./CacheTypeSelector";

const renderStandard = (overrides: Partial<React.ComponentProps<typeof CacheTypeSelector>> = {}) =>
  render(
    <CacheTypeSelector
      cacheMode="standard"
      deploymentType="node"
      onCacheModeChange={vi.fn()}
      onDeploymentTypeChange={vi.fn()}
      {...overrides}
    />,
  );

const openSelectFor = (labelText: string) => {
  const selector = screen.getByText(labelText).parentElement!.querySelector(".ant-select-selector");
  fireEvent.mouseDown(selector as Element);
};

const optionLabels = async (): Promise<string[]> =>
  waitFor(() => {
    const options = Array.from(document.querySelectorAll(".ant-select-item-option-content"));
    expect(options.length).toBeGreaterThan(0);
    return options.map((el) => el.textContent ?? "");
  });

describe("CacheTypeSelector", () => {
  it("should not offer Semantic as a Redis deployment type", async () => {
    renderStandard();

    openSelectFor("Redis Deployment Type");

    expect(await optionLabels()).toEqual(["Node (Single Instance)", "Cluster", "Sentinel"]);
  });

  it("should offer Semantic only as a Cache Type", async () => {
    renderStandard();

    openSelectFor("Cache Type");

    expect(await optionLabels()).toEqual(["Standard (exact match)", "Semantic (similarity-based)"]);
  });

  it("should call onCacheModeChange with semantic when the semantic cache type is chosen", async () => {
    const onCacheModeChange = vi.fn();
    renderStandard({ onCacheModeChange });

    openSelectFor("Cache Type");
    const semantic = await waitFor(() => {
      const match = Array.from(document.querySelectorAll(".ant-select-item-option")).find((el) =>
        el.textContent?.includes("Semantic"),
      );
      expect(match).toBeTruthy();
      return match as HTMLElement;
    });
    fireEvent.click(semantic);

    expect(onCacheModeChange).toHaveBeenCalledWith("semantic");
  });

  it("should hide the deployment type selector when semantic caching is selected", () => {
    renderStandard({ cacheMode: "semantic" });

    expect(screen.queryByText("Redis Deployment Type")).not.toBeInTheDocument();
  });
});
