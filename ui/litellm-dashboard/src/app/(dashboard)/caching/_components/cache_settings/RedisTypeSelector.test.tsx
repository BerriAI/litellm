import { describe, expect, it, vi } from "vitest";
import RedisTypeSelector from "./RedisTypeSelector";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

describe("RedisTypeSelector", () => {
  it("should render the component", () => {
    const { getAllByText } = render(
      <RedisTypeSelector redisType="redis" redisTypeDescriptions={{}} onTypeChange={() => {}} />,
    );
    expect(getAllByText(/Redis/i).length).toBeGreaterThan(0);
  });

  it("should keep the deployment topologies selectable when config.yaml only fixes the semantic choice", async () => {
    const user = userEvent.setup();
    const onTypeChange = vi.fn();
    render(
      <RedisTypeSelector
        redisType="node"
        redisTypeDescriptions={{}}
        onTypeChange={onTypeChange}
        unavailableTypes={new Set(["semantic"])}
      />,
    );

    await user.click(screen.getByLabelText("Redis Type"));

    expect(await screen.findByRole("option", { name: "Semantic" })).toHaveAttribute("data-disabled");
    expect(screen.getByRole("option", { name: "Cluster" })).not.toHaveAttribute("data-disabled");

    await user.click(screen.getByRole("option", { name: "Cluster" }));
    expect(onTypeChange).toHaveBeenCalledWith("cluster");
  });
});
