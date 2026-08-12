import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PromptCodeSnippets from "./PromptCodeSnippets";

describe("PromptCodeSnippets", () => {
  it("opens generated code for the selected prompt", async () => {
    render(
      <PromptCodeSnippets
        promptId="welcome"
        model="gpt-4o"
        promptVariables={{ name: "Ada" }}
        accessToken="token"
        version="2"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /get code/i }));
    expect(await screen.findByText("Generated Code")).toBeInTheDocument();
    expect(screen.getByText(/welcome/)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Language" })).toBeInTheDocument();
    expect(screen.getByRole("tablist", { name: "Generated code type" })).toBeInTheDocument();
    expect(screen.getByRole("dialog")).toHaveClass("max-h-[calc(100dvh-2rem)]", "overflow-y-auto");
  });
});
