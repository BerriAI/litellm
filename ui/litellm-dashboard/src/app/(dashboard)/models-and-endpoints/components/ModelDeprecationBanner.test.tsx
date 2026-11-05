/* @vitest-environment jsdom */
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  useModelDeprecations,
  type ModelDeprecationInfo,
  type ModelDeprecationResponse,
} from "@/app/(dashboard)/hooks/models/useModelDeprecations";
import { uiHref } from "@/utils/uiHref";
import ModelDeprecationBanner from "./ModelDeprecationBanner";

vi.mock("@/app/(dashboard)/hooks/models/useModelDeprecations", () => ({
  useModelDeprecations: vi.fn(),
}));

const mockedUseModelDeprecations = vi.mocked(useModelDeprecations);

const makeResponse = (overrides: Partial<ModelDeprecationResponse> = {}): ModelDeprecationResponse => ({
  checked_at: "2026-11-04T00:00:00Z",
  warn_within_days: 30,
  deprecated: [],
  imminent: [],
  upcoming: [],
  ...overrides,
});

const makeInfo = (overrides: Partial<ModelDeprecationInfo> = {}): ModelDeprecationInfo => ({
  model_name: "gpt-4o",
  deprecation_date: "2026-11-30",
  days_until_deprecation: 26,
  status: "imminent",
  ...overrides,
});

const queryResult = (
  data: ModelDeprecationResponse | undefined,
  isLoading = false,
  isError = false,
): ReturnType<typeof useModelDeprecations> => ({ data, isLoading, isError }) as ReturnType<typeof useModelDeprecations>;

const pinnedToday = (): Date => new Date(2026, 10, 4);

const renderBanner = (today = pinnedToday()) => render(<ModelDeprecationBanner today={today} />);

describe("ModelDeprecationBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
  });

  it("renders nothing when there are no deprecated or imminent models", () => {
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse()));

    renderBanner();

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("renders nothing when only upcoming models are present", () => {
    mockedUseModelDeprecations.mockReturnValue(
      queryResult(
        makeResponse({
          upcoming: [makeInfo({ status: "upcoming" })],
        }),
      ),
    );

    renderBanner();

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("renders nothing while loading or after an error", () => {
    mockedUseModelDeprecations.mockReturnValue(queryResult(undefined, true));
    const { rerender } = renderBanner();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    mockedUseModelDeprecations.mockReturnValue(queryResult(undefined, false, true));
    rerender(<ModelDeprecationBanner today={pinnedToday()} />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows deprecated and imminent entries with a link to Logging & Alerts", () => {
    const deprecatedInfo: ModelDeprecationInfo = {
      model_name: "gpt-4",
      team_public_model_name: null,
      litellm_provider: "openai",
      deprecation_date: "2026-06-01",
      days_until_deprecation: -152,
      status: "deprecated",
      successor_model: "gpt-5.6-sol",
    };
    const imminentInfo: ModelDeprecationInfo = {
      model_name: "gpt-4o",
      team_public_model_name: null,
      litellm_provider: "openai",
      deprecation_date: "2026-11-30",
      days_until_deprecation: 26,
      status: "imminent",
      successor_model: "X",
    };
    const response = makeResponse({
      deprecated: [deprecatedInfo],
      imminent: [imminentInfo],
    });
    mockedUseModelDeprecations.mockReturnValue(queryResult(response));

    renderBanner();

    expect(screen.getByText("1 model is deprecated, 1 deprecates within 30 days")).toBeInTheDocument();
    const entries = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(entries[0]).toHaveTextContent("gpt-4 (openai) was deprecated on 2026-06-01, migrate to gpt-5.6-sol");
    expect(entries[1]).toHaveTextContent("gpt-4o (openai) deprecates on 2026-11-30, in 26 days, migrate to X");
    expect(screen.getByRole("link", { name: "Get Slack alerts for deprecations" })).toHaveAttribute(
      "href",
      `${uiHref("logging-and-alerts")}?tab=alerting-types`,
    );
  });

  it("treats a server-deprecated entry dated today as imminent locally", () => {
    const infoOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "due-today",
      deprecation_date: "2026-11-04",
      days_until_deprecation: -1,
      status: "deprecated",
    };
    const info = makeInfo(infoOverrides);
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ deprecated: [info] })));

    renderBanner();

    expect(screen.getByText("1 model deprecates within 30 days")).toBeInTheDocument();
    expect(within(screen.getByRole("list")).getByRole("listitem")).toHaveTextContent(
      "due-today deprecates today (2026-11-04)",
    );
  });

  it("treats a server-imminent entry dated yesterday as deprecated locally", () => {
    const infoOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "due-yesterday",
      deprecation_date: "2026-11-03",
      days_until_deprecation: 0,
      status: "imminent",
    };
    const info = makeInfo(infoOverrides);
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ imminent: [info] })));

    renderBanner();

    expect(screen.getByText("1 model is deprecated")).toBeInTheDocument();
    expect(within(screen.getByRole("list")).getByRole("listitem")).toHaveTextContent(
      "due-yesterday was deprecated on 2026-11-03",
    );
  });

  it("uses the local date difference instead of the server day count", () => {
    const infoOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "due-in-two",
      deprecation_date: "2026-11-06",
      days_until_deprecation: 1,
      status: "imminent",
    };
    const info = makeInfo(infoOverrides);
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ imminent: [info] })));

    renderBanner();

    expect(within(screen.getByRole("list")).getByRole("listitem")).toHaveTextContent(
      "due-in-two deprecates on 2026-11-06, in 2 days",
    );
  });

  it("deduplicates entries by their displayed name, provider, and deprecation date", () => {
    const globalInfoOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "o4-mini",
      team_public_model_name: null,
      litellm_provider: "openai",
      deprecation_date: "2026-10-23",
      status: "deprecated",
    };
    const firstTeamInfoOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "o4-mini-team-a",
      team_public_model_name: "o4-mini",
      litellm_provider: "openai",
      deprecation_date: "2026-10-23",
      status: "deprecated",
    };
    const secondTeamInfoOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "o4-mini-team-b",
      team_public_model_name: "o4-mini",
      litellm_provider: "openai",
      deprecation_date: "2026-10-23",
      status: "deprecated",
    };
    mockedUseModelDeprecations.mockReturnValue(
      queryResult(
        makeResponse({
          deprecated: [
            makeInfo(globalInfoOverrides),
            makeInfo(firstTeamInfoOverrides),
            makeInfo(secondTeamInfoOverrides),
          ],
        }),
      ),
    );

    renderBanner();

    expect(screen.getByText("1 model is deprecated")).toBeInTheDocument();
    const entries = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(entries).toHaveLength(1);
    expect(entries[0]).toHaveTextContent("o4-mini (openai) was deprecated on 2026-10-23");
  });

  it("keeps entries with different successors and renders each migration target", () => {
    const firstEntryOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "team-model-a",
      team_public_model_name: "shared-model",
      litellm_provider: "openai",
      deprecation_date: "2026-10-23",
      days_until_deprecation: -1,
      status: "deprecated",
      successor_model: "successor-a",
    };
    const secondEntryOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "shared-model",
      litellm_provider: "openai",
      deprecation_date: "2026-10-23",
      days_until_deprecation: -1,
      status: "deprecated",
      successor_model: "successor-b",
    };
    const firstEntry = makeInfo(firstEntryOverrides);
    const secondEntry = makeInfo(secondEntryOverrides);
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ deprecated: [firstEntry, secondEntry] })));

    renderBanner();

    const entries = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(screen.getByText("1 model is deprecated")).toBeInTheDocument();
    expect(entries).toHaveLength(2);
    expect(entries[0]).toHaveTextContent("shared-model (openai) was deprecated on 2026-10-23, migrate to successor-a");
    expect(entries[1]).toHaveTextContent("shared-model (openai) was deprecated on 2026-10-23, migrate to successor-b");
  });

  it("keeps entries with different deprecation dates", () => {
    const firstEntryOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "shared-model",
      litellm_provider: "openai",
      deprecation_date: "2026-10-22",
      days_until_deprecation: -2,
      status: "deprecated",
      successor_model: "successor",
    };
    const secondEntryOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "shared-model",
      litellm_provider: "openai",
      deprecation_date: "2026-10-23",
      days_until_deprecation: -1,
      status: "deprecated",
      successor_model: "successor",
    };
    const firstEntry = makeInfo(firstEntryOverrides);
    const secondEntry = makeInfo(secondEntryOverrides);
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ deprecated: [firstEntry, secondEntry] })));

    renderBanner();

    const entries = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(entries).toHaveLength(2);
    expect(entries[0]).toHaveTextContent("shared-model (openai) was deprecated on 2026-10-22");
    expect(entries[1]).toHaveTextContent("shared-model (openai) was deprecated on 2026-10-23");
  });

  it("dismisses the banner and keeps it dismissed after remounting the same entries", async () => {
    const info = makeInfo();
    const secondInfo = makeInfo({ model_name: "claude-sonnet-4" });
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ imminent: [info, secondInfo] })));
    const user = userEvent.setup();
    const { unmount } = renderBanner();

    await user.click(screen.getByRole("button", { name: "Dismiss deprecation warning" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    unmount();

    const changedSecondInfoOverrides: Partial<ModelDeprecationInfo> = {
      ...secondInfo,
      days_until_deprecation: secondInfo.days_until_deprecation - 1,
    };
    const changedInfoOverrides: Partial<ModelDeprecationInfo> = {
      ...info,
      days_until_deprecation: info.days_until_deprecation - 1,
    };
    const teamAliasInfoOverrides: Partial<ModelDeprecationInfo> = {
      ...info,
      model_name: "team-internal-name",
      team_public_model_name: info.model_name,
      days_until_deprecation: info.days_until_deprecation - 1,
    };
    mockedUseModelDeprecations.mockReturnValue(
      queryResult(
        makeResponse({
          imminent: [
            makeInfo(changedSecondInfoOverrides),
            makeInfo(changedInfoOverrides),
            makeInfo(teamAliasInfoOverrides),
          ],
        }),
      ),
    );
    renderBanner();

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the banner again when a new entry is added after dismissal", async () => {
    const firstEntryOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "gpt-4",
      deprecation_date: "2026-10-23",
      days_until_deprecation: -12,
      status: "deprecated",
    };
    const firstEntry = makeInfo(firstEntryOverrides);
    const addedEntryOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "gpt-3.5",
      deprecation_date: "2026-10-23",
      days_until_deprecation: -12,
      status: "deprecated",
    };
    const addedEntry = makeInfo(addedEntryOverrides);
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ deprecated: [firstEntry] })));
    const user = userEvent.setup();
    const { rerender } = renderBanner();

    await user.click(screen.getByRole("button", { name: "Dismiss deprecation warning" }));
    mockedUseModelDeprecations.mockReturnValue(
      queryResult(
        makeResponse({
          deprecated: [firstEntry, addedEntry],
        }),
      ),
    );
    rerender(<ModelDeprecationBanner today={pinnedToday()} />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("2 models are deprecated")).toBeInTheDocument();
  });

  it("shows the banner again when an imminent entry becomes deprecated", async () => {
    const imminentEntryOverrides: Partial<ModelDeprecationInfo> = {
      model_name: "gpt-4o",
      deprecation_date: "2026-11-30",
      days_until_deprecation: 0,
      status: "imminent",
    };
    const imminentEntry = makeInfo(imminentEntryOverrides);
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ imminent: [imminentEntry] })));
    const user = userEvent.setup();
    const { rerender } = renderBanner(new Date(2026, 10, 30));

    await user.click(screen.getByRole("button", { name: "Dismiss deprecation warning" }));
    rerender(<ModelDeprecationBanner today={new Date(2026, 11, 1)} />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("1 model is deprecated")).toBeInTheDocument();
  });

  it("shows the banner again when an entry's successor changes after dismissal", async () => {
    const entry = makeInfo({ successor_model: "successor-a" });
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ imminent: [entry] })));
    const user = userEvent.setup();
    const { rerender } = renderBanner();

    await user.click(screen.getByRole("button", { name: "Dismiss deprecation warning" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    mockedUseModelDeprecations.mockReturnValue(
      queryResult(makeResponse({ imminent: [makeInfo({ ...entry, successor_model: "successor-b" })] })),
    );
    rerender(<ModelDeprecationBanner today={pinnedToday()} />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/migrate to successor-b/)).toBeInTheDocument();
  });

  it("expands all entries and collapses back to the first five", async () => {
    const imminent = [
      ...Array.from({ length: 7 }, (_value, index) => makeInfo({ model_name: `model-${index + 1}` })),
      makeInfo({ model_name: "team-model-1", team_public_model_name: "model-1" }),
    ];
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ imminent })));
    const user = userEvent.setup();

    renderBanner();

    const entries = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(entries).toHaveLength(5);
    expect(entries[4]).toHaveTextContent("model-5 deprecates");
    expect(screen.queryByText(/model-6/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show all 7" }));
    expect(within(screen.getByRole("list")).getAllByRole("listitem")).toHaveLength(7);
    expect(screen.getByText(/model-6 deprecates/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show fewer" }));
    expect(within(screen.getByRole("list")).getAllByRole("listitem")).toHaveLength(5);
    expect(screen.queryByText(/model-6/)).not.toBeInTheDocument();
  });
});
