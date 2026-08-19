import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import CacheControlInjectionPoints, { type CacheControlInjectionPoint } from "./cache_control_settings";

const ROLE_HINT = "LiteLLM will mark all messages of this role as cacheable";
const INDEX_HINT = "(Optional) If set litellm will mark the message at this index as cacheable";

const ONE_POINT: CacheControlInjectionPoint[] = [{ location: "message" }];

const hintTrigger = (fieldLabel: string): Element => {
  const label = screen.getByText(fieldLabel);
  const trigger = label.parentElement?.querySelector('[aria-label="question-circle"]');
  if (!(trigger instanceof Element)) {
    throw new Error(`no hint trigger beside the ${fieldLabel} label`);
  }
  return trigger;
};

describe("CacheControlInjectionPoints field hints", () => {
  it("explains on the Role field that the role marks every message of that role cacheable", async () => {
    const user = userEvent.setup();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    await user.hover(hintTrigger("Role"));

    expect(await screen.findByText(ROLE_HINT)).toBeInTheDocument();
  });

  it("explains on the Index field that it is optional and marks that message cacheable", async () => {
    const user = userEvent.setup();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    await user.hover(hintTrigger("Index"));

    expect(await screen.findByText(INDEX_HINT)).toBeInTheDocument();
  });

  it("keeps both hints behind a hover rather than rendering them inline", () => {
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    expect(screen.queryByText(ROLE_HINT)).not.toBeInTheDocument();
    expect(screen.queryByText(INDEX_HINT)).not.toBeInTheDocument();
  });

  it("gives every row its own pair of hints", () => {
    render(
      <CacheControlInjectionPoints value={[{ location: "message" }, { location: "message" }]} onChange={vi.fn()} />,
    );

    expect(screen.getAllByText("Role")).toHaveLength(2);
    expect(screen.getAllByText("Index")).toHaveLength(2);
    expect(screen.getAllByLabelText("question-circle")).toHaveLength(4);
  });

  it("still reports a typed index as a string through onChange", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={onChange} />);

    await user.type(screen.getByPlaceholderText("Optional"), "3");

    expect(onChange).toHaveBeenLastCalledWith([{ location: "message", index: "3" }]);
  });
});
