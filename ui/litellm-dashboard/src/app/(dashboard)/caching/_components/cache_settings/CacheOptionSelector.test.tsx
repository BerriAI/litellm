import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CacheOptionSelector from "./CacheOptionSelector";

const OPTIONS = [
  { value: "node", label: "Node (Single Instance)" },
  { value: "cluster", label: "Cluster" },
] as const;

const DESCRIPTIONS = { node: "single instance", cluster: "cluster mode" };

describe("CacheOptionSelector", () => {
  it("should show the description for the selected value", () => {
    render(
      <CacheOptionSelector
        label="Redis Type"
        value="cluster"
        options={OPTIONS}
        descriptions={DESCRIPTIONS}
        fallbackDescription="fallback"
        onChange={() => {}}
      />,
    );
    expect(screen.getByText("cluster mode")).toBeInTheDocument();
    expect(screen.getByText("Redis Type")).toBeInTheDocument();
  });

  it("should emit the chosen option value on selection", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <CacheOptionSelector
        label="Redis Type"
        value="node"
        options={OPTIONS}
        descriptions={DESCRIPTIONS}
        fallbackDescription="fallback"
        onChange={onChange}
      />,
    );

    await user.click(document.querySelector(".ant-select-selector") as HTMLElement);
    await user.click(await screen.findByText("Cluster"));

    expect(onChange).toHaveBeenCalledWith("cluster");
  });
});
