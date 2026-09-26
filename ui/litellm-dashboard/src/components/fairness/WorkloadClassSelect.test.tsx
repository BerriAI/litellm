import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { useShowWorkloadClass, WorkloadClassSelect } from "./WorkloadClassSelect";

const state = { userRole: "Admin", configuredClasses: [] as string[] };
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ userRole: state.userRole }) }));
vi.mock("@/lib/http/api", () => ({
  fetchClient: {
    GET: vi.fn(async () => ({
      data: { settings: { workload_classes: state.configuredClasses.map((name) => ({ name })) } },
    })),
  },
}));

const withClient = (ui: React.ReactElement) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {ui}
    </QueryClientProvider>,
  );

const ShowProbe = ({ current }: { current: string | undefined }) => {
  const show = useShowWorkloadClass(current);
  return <span data-testid="show">{show ? "shown" : "hidden"}</span>;
};

describe("WorkloadClassSelect", () => {
  it("lists the default pool plus configured classes and reports the picked class", async () => {
    const onChange = vi.fn();
    const fetchNames = vi.fn().mockResolvedValue(["production", "batch"]);
    withClient(<WorkloadClassSelect value={undefined} onChange={onChange} fetchNames={fetchNames} />);

    const trigger = screen.getByRole("combobox", { name: "Workload class" });
    expect(trigger).toHaveTextContent("default");

    await userEvent.click(trigger);
    await screen.findByRole("option", { name: "production" });
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual([
      "default (shared pool)",
      "production",
      "batch",
    ]);

    await userEvent.click(screen.getByRole("option", { name: "batch" }));
    expect(onChange).toHaveBeenCalledWith("batch");
  });

  it("keeps a stored class selectable even when it is no longer configured", async () => {
    const fetchNames = vi.fn().mockResolvedValue(["production"]);
    withClient(<WorkloadClassSelect value="legacy" onChange={vi.fn()} fetchNames={fetchNames} />);

    const trigger = screen.getByRole("combobox", { name: "Workload class" });
    expect(trigger).toHaveTextContent("legacy");
    await userEvent.click(trigger);
    expect(await screen.findByRole("option", { name: "legacy" })).toBeInTheDocument();
  });
});

describe("useShowWorkloadClass", () => {
  it("hides the field when no classes are configured and the value is the default pool", () => {
    withClient(<ShowProbe current="default" />);
    expect(screen.getByTestId("show")).toHaveTextContent("hidden");
  });

  it("shows the field when the stored value is a named class even without configured classes", () => {
    withClient(<ShowProbe current="batch" />);
    expect(screen.getByTestId("show")).toHaveTextContent("shown");
  });

  it("shows the field once the proxy reports configured classes", async () => {
    state.configuredClasses = ["production"];
    withClient(<ShowProbe current={undefined} />);
    expect(await screen.findByText("shown")).toBeInTheDocument();
  });
});
