import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption } from "@/../tests/test-utils";
import { referenceRatesKeys } from "@/app/(dashboard)/hooks/referenceRates/useReferenceRates";
import { registerErrorHandler } from "@/lib/http/runtime";
import SpendDisplayCurrency from "./SpendDisplayCurrency";

const RATES = { source: "ecb", rate_date: "2026-11-13", usd_per_unit: { USD: 1, EUR: 1.25, JPY: 0.01 } };

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const newClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

const renderFresh = (ui: ReactElement, client = newClient()) => {
  const wrap = (node: ReactElement) => <QueryClientProvider client={client}>{node}</QueryClientProvider>;
  const rendered = render(wrap(ui));
  return { ...rendered, client, rerender: (next: ReactElement) => rendered.rerender(wrap(next)) };
};

const trigger = () => screen.getByRole("combobox", { name: "Spend display" });

const UNAVAILABLE = "Reference rates are unavailable, spend is shown in USD";

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date("2026-11-14T12:00:00Z"));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  registerErrorHandler(() => {});
});

describe("SpendDisplayCurrency", () => {
  it("stays on USD without asking for rates until the selector is opened", () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />);

    expect(trigger()).toHaveTextContent("USD");
    expect(screen.queryByText(/estimated/)).not.toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("estimates the current range total in the chosen currency and follows later totals", async () => {
    const user = userEvent.setup();
    const fetch = vi.fn().mockResolvedValue(json(RATES));
    vi.stubGlobal("fetch", fetch);
    const { rerender } = renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />);

    await chooseSelectOption(user, trigger(), /^EUR/);
    expect(await screen.findByText("≈ EUR 34.00 estimated at the ECB reference rate of 2026-11-13")).toBeVisible();

    rerender(<SpendDisplayCurrency usdSpend={50} loading={false} />);
    expect(screen.getByText("≈ EUR 40.00 estimated at the ECB reference rate of 2026-11-13")).toBeVisible();

    await chooseSelectOption(user, trigger(), /^JPY/);
    expect(screen.getByText("≈ JPY 5,000 estimated at the ECB reference rate of 2026-11-13")).toBeVisible();

    await chooseSelectOption(user, trigger(), /^USD/);
    expect(screen.queryByText(/estimated/)).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("keeps USD with a notice and no dashboard error when rates are unavailable", async () => {
    const user = userEvent.setup();
    const onError = vi.fn();
    registerErrorHandler(onError);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "ECB reference rates unavailable" }, 503)));
    renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />);

    await user.click(trigger());

    expect(await screen.findByText("Reference rates are unavailable, spend is shown in USD")).toBeVisible();
    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["USD · US Dollar"]);
    expect(onError).not.toHaveBeenCalled();
  });

  it("falls back to USD when a refetch fails after another currency was chosen", async () => {
    const user = userEvent.setup();
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(json(RATES))
      .mockResolvedValue(json({ detail: "unavailable" }, 503));
    vi.stubGlobal("fetch", fetch);
    const { client } = renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />);
    await chooseSelectOption(user, trigger(), /^EUR/);
    expect(await screen.findByText(/≈ EUR 34\.00 estimated/)).toBeVisible();

    await act(() => client.invalidateQueries({ queryKey: referenceRatesKeys.all }));

    expect(await screen.findByText("Reference rates are unavailable, spend is shown in USD")).toBeVisible();
    expect(screen.queryByText(/estimated/)).not.toBeInTheDocument();
    expect(trigger()).toHaveTextContent("USD");
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("asks for rates again when the selector is reopened after they failed", async () => {
    const user = userEvent.setup();
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(json({ detail: "unavailable" }, 503))
      .mockResolvedValue(json(RATES));
    vi.stubGlobal("fetch", fetch);
    renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />);
    await user.click(trigger());
    expect(await screen.findByText(UNAVAILABLE)).toBeVisible();
    await user.keyboard("{Escape}");

    await chooseSelectOption(user, trigger(), /^EUR/);

    expect(await screen.findByText(/≈ EUR 34\.00 estimated/)).toBeVisible();
    expect(screen.queryByText(UNAVAILABLE)).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("keeps a failure cached by another card hidden until this card is opened", async () => {
    const user = userEvent.setup();
    const fetch = vi.fn().mockResolvedValue(json({ detail: "unavailable" }, 503));
    vi.stubGlobal("fetch", fetch);
    const client = newClient();
    const first = renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />, client);
    await user.click(trigger());
    expect(await screen.findByText(UNAVAILABLE)).toBeVisible();
    first.unmount();

    renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />, client);

    expect(screen.queryByText(UNAVAILABLE)).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
    await user.click(trigger());
    expect(await screen.findByText(UNAVAILABLE)).toBeVisible();
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  });

  it("cancels the rate request when the overview goes away before it answers", async () => {
    const user = userEvent.setup();
    const requests: Request[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((request: Request) => {
        requests.push(request);
        return new Promise<Response>(() => {});
      }),
    );
    const { unmount } = renderFresh(<SpendDisplayCurrency usdSpend={42.5} loading={false} />);

    await user.click(trigger());
    await waitFor(() => expect(requests).toHaveLength(1));
    unmount();

    expect(requests[0].signal.aborted).toBe(true);
  });
});
