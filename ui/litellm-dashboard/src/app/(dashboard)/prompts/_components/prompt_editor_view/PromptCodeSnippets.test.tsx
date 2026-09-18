import { fireEvent, render, screen } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
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

  it("shows the selected language by its human label on the trigger", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
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

    const trigger = await screen.findByRole("combobox", { name: "Language" });
    expect(trigger).toHaveTextContent("cURL");

    await user.click(trigger);
    const python = await screen.findByRole("option", { name: "Python (OpenAI SDK)" });
    await user.click(python);

    expect(screen.getByRole("combobox", { name: "Language" })).toHaveTextContent("Python (OpenAI SDK)");
  });

  it("includes the viewed environment in every generated request", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    render(
      <PromptCodeSnippets
        promptId="welcome"
        model="gpt-4o"
        accessToken="token"
        version="2"
        environment="development"
      />,
    );
    await user.click(screen.getByRole("button", { name: /get code/i }));
    await screen.findByText("Generated Code");

    await user.click(screen.getByRole("button", { name: /copy to clipboard/i }));
    expect(await navigator.clipboard.readText()).toContain('"prompt_environment": "development"');

    await user.click(screen.getByRole("tab", { name: "With Version" }));
    await user.click(screen.getByRole("button", { name: /copy to clipboard/i }));
    const versionSnippet = await navigator.clipboard.readText();
    expect(versionSnippet).toContain('"prompt_environment": "development"');
    expect(versionSnippet).toContain('"prompt_version": 2');
  });
});
