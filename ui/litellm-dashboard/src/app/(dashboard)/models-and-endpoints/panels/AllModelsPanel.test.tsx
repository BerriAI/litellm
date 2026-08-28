/* @vitest-environment jsdom */
import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AllModelsPanel from "./AllModelsPanel";

interface CapturedTabProps {
  selectedModelGroup: string | null;
  setSelectedModelGroup: (modelGroup: string) => void;
  selectedModelAccessGroupFilter: string | null;
  setSelectedModelAccessGroupFilter: (accessGroup: string | null) => void;
}

const captured: { props: CapturedTabProps | null } = { props: null };

vi.mock("@/app/(dashboard)/models-and-endpoints/components/AllModelsTab", () => ({
  default: (props: CapturedTabProps) => {
    captured.props = props;
    return <div data-testid="all-models-tab" />;
  },
}));
vi.mock("@/app/(dashboard)/models-and-endpoints/components/AllModelsTable", () => ({
  ALL_MODEL_GROUPS_VALUE: "all",
}));
vi.mock("@/app/(dashboard)/models-and-endpoints/useModelDashboardData", () => ({
  useModelDashboardData: () => ({ availableModelGroups: [], availableModelAccessGroups: [] }),
}));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));

describe("AllModelsPanel ?model_group= filter", () => {
  beforeEach(() => {
    captured.props = null;
    window.history.pushState(null, "", "/models-and-endpoints/");
  });

  it("seeds the model group filter from the URL", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model_group=gpt-4o");
    render(<AllModelsPanel />);
    expect(captured.props?.selectedModelGroup).toBe("gpt-4o");
  });

  it("defaults to no filter when the param is absent", () => {
    render(<AllModelsPanel />);
    expect(captured.props?.selectedModelGroup).toBeNull();
  });

  it("writes the selected group to the URL via replaceState", () => {
    render(<AllModelsPanel />);
    const spy = vi.spyOn(window.history, "replaceState");
    act(() => captured.props?.setSelectedModelGroup("claude-opus-5"));
    expect(spy.mock.calls.at(-1)?.[2]).toContain("model_group=claude-opus-5");
    spy.mockRestore();
  });

  it("removes the param when the filter resets to all groups", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model_group=gpt-4o");
    render(<AllModelsPanel />);
    const spy = vi.spyOn(window.history, "replaceState");
    act(() => captured.props?.setSelectedModelGroup("all"));
    expect(spy.mock.calls.at(-1)?.[2] as string).not.toContain("model_group");
    spy.mockRestore();
  });
});

describe("AllModelsPanel ?model_access_group= filter", () => {
  beforeEach(() => {
    captured.props = null;
    window.history.pushState(null, "", "/models-and-endpoints/");
  });

  it("seeds the access group filter from the URL", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model_access_group=sales-team");
    render(<AllModelsPanel />);
    expect(captured.props?.selectedModelAccessGroupFilter).toBe("sales-team");
  });

  it("writes the selected access group to the URL via replaceState", () => {
    render(<AllModelsPanel />);
    const spy = vi.spyOn(window.history, "replaceState");
    act(() => captured.props?.setSelectedModelAccessGroupFilter("sales-team"));
    expect(spy.mock.calls.at(-1)?.[2]).toContain("model_access_group=sales-team");
    spy.mockRestore();
  });

  it("removes the param when the access group filter clears", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model_access_group=sales-team");
    render(<AllModelsPanel />);
    const spy = vi.spyOn(window.history, "replaceState");
    act(() => captured.props?.setSelectedModelAccessGroupFilter(null));
    expect(spy.mock.calls.at(-1)?.[2] as string).not.toContain("model_access_group");
    spy.mockRestore();
  });

  it("the two filter params stay independent", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model_group=gpt-4o&model_access_group=sales-team");
    render(<AllModelsPanel />);
    expect(captured.props?.selectedModelGroup).toBe("gpt-4o");
    expect(captured.props?.selectedModelAccessGroupFilter).toBe("sales-team");

    const spy = vi.spyOn(window.history, "replaceState");
    act(() => captured.props?.setSelectedModelAccessGroupFilter(null));
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("model_group=gpt-4o");
    expect(url).not.toContain("model_access_group");
    spy.mockRestore();
  });
});
