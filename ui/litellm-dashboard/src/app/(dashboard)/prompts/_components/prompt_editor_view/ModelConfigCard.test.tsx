import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ModelConfigCard from "./ModelConfigCard";

vi.mock("@/components/common_components/ModelSelector", () => ({
  default: ({ value }: any) => <div>Model: {value}</div>,
}));

describe("ModelConfigCard", () => {
  it("edits model parameters", () => {
    const onTemperatureChange = vi.fn();
    const onMaxTokensChange = vi.fn();
    render(
      <ModelConfigCard
        model="gpt-4o"
        accessToken="token"
        onModelChange={vi.fn()}
        temperature={1}
        maxTokens={1000}
        onTemperatureChange={onTemperatureChange}
        onMaxTokensChange={onMaxTokensChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Parameters" }));
    fireEvent.change(screen.getByDisplayValue("1"), { target: { value: "0.4" } });
    fireEvent.change(screen.getByDisplayValue("1000"), { target: { value: "2048" } });
    expect(onTemperatureChange).toHaveBeenCalledWith(0.4);
    expect(onMaxTokensChange).toHaveBeenCalledWith(2048);
  });
});
