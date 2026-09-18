import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { UnifiedSelector } from "./UnifiedSelector";
import { EndpointId, ENDPOINT_CONFIGS } from "../endpoint_config";

const CHAT = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];
const AGENTS = ENDPOINT_CONFIGS[EndpointId.A2A_AGENTS];

const promptIsVisible = (text: string) =>
  screen.queryByText(text) !== null || screen.queryByPlaceholderText(text) !== null;

const openList = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("combobox"));
};

describe("UnifiedSelector", () => {
  it("renders a combobox", () => {
    render(
      <UnifiedSelector
        value=""
        options={[{ value: "option1", label: "Option One" }]}
        loading={false}
        config={CHAT}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("prompts with the endpoint's own selector copy", () => {
    render(<UnifiedSelector value="" options={[]} loading={false} config={CHAT} onChange={vi.fn()} />);

    expect(promptIsVisible(CHAT.selectorPlaceholder)).toBe(true);
  });

  it("prompts with the agent endpoint's copy when configured for agents", () => {
    render(<UnifiedSelector value="" options={[]} loading={false} config={AGENTS} onChange={vi.fn()} />);

    expect(promptIsVisible(AGENTS.selectorPlaceholder)).toBe(true);
  });

  it("swaps the prompt for a loading message while options are in flight", () => {
    render(<UnifiedSelector value="" options={[]} loading config={CHAT} onChange={vi.fn()} />);

    expect(promptIsVisible(`Loading ${CHAT.selectorLabel.toLowerCase()}s...`)).toBe(true);
  });

  it("reports the chosen option's value, not its label", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <UnifiedSelector
        value=""
        options={[
          { value: "option1", label: "Option One" },
          { value: "option2", label: "Option Two" },
        ]}
        loading={false}
        config={CHAT}
        onChange={onChange}
      />,
    );

    await openList(user);

    const matches = await screen.findAllByText("Option Two");
    await user.click(matches[matches.length - 1]);

    await waitFor(() => expect(onChange).toHaveBeenCalled());
    expect(onChange.mock.calls[0][0]).toBe("option2");
  });

  it("narrows the list as the user searches", async () => {
    const user = userEvent.setup();
    render(
      <UnifiedSelector
        value=""
        options={[
          { value: "option1", label: "Option One" },
          { value: "option2", label: "Option Two" },
          { value: "option3", label: "Different" },
        ]}
        loading={false}
        config={CHAT}
        onChange={vi.fn()}
      />,
    );

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);
    fireEvent.change(combobox, { target: { value: "One" } });

    await waitFor(() => {
      expect(screen.getAllByText("Option One").length).toBeGreaterThan(0);
      expect(screen.queryByText("Option Two")).not.toBeInTheDocument();
      expect(screen.queryByText("Different")).not.toBeInTheDocument();
    });
  });

  it("shows a busy indicator in the empty list while loading", async () => {
    const user = userEvent.setup();
    render(<UnifiedSelector value="" options={[]} loading config={CHAT} onChange={vi.fn()} />);

    await openList(user);

    await waitFor(() => {
      expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument();
    });
  });

  it("says so when there is nothing to pick and nothing is loading", async () => {
    const user = userEvent.setup();
    render(<UnifiedSelector value="" options={[]} loading={false} config={CHAT} onChange={vi.fn()} />);

    await openList(user);

    expect(await screen.findByText(`No ${CHAT.selectorLabel.toLowerCase()}s available`)).toBeInTheDocument();
  });
});
