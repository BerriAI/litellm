/* @vitest-environment jsdom */
import { render, screen, within } from "@testing-library/react";
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
): ReturnType<typeof useModelDeprecations> =>
  ({ data, isLoading, isError }) as ReturnType<typeof useModelDeprecations>;

describe("ModelDeprecationBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when there are no deprecated or imminent models", () => {
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse()));

    render(<ModelDeprecationBanner />);

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

    render(<ModelDeprecationBanner />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("renders nothing while loading or after an error", () => {
    mockedUseModelDeprecations.mockReturnValue(queryResult(undefined, true));
    const { rerender } = render(<ModelDeprecationBanner />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    mockedUseModelDeprecations.mockReturnValue(queryResult(undefined, false, true));
    rerender(<ModelDeprecationBanner />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows deprecated and imminent entries with a link to Logging & Alerts", () => {
    const deprecatedInfo: ModelDeprecationInfo = {
      model_name: "gpt-4",
      litellm_provider: "openai",
      deprecation_date: "2026-06-01",
      days_until_deprecation: -152,
      status: "deprecated",
      successor_model: "gpt-5.6-sol",
    };
    const imminentInfo: ModelDeprecationInfo = {
      model_name: "gpt-4o",
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

    render(<ModelDeprecationBanner />);

    expect(screen.getByText("1 model is deprecated, 1 deprecates within 30 days")).toBeInTheDocument();
    const entries = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(entries[0]).toHaveTextContent("gpt-4 (openai) was deprecated on 2026-06-01, migrate to gpt-5.6-sol");
    expect(entries[1]).toHaveTextContent("gpt-4o (openai) deprecates on 2026-11-30, in 26 days, migrate to X");
    expect(screen.getByRole("link", { name: "Get Slack alerts for deprecations" })).toHaveAttribute(
      "href",
      uiHref("logging-and-alerts"),
    );
  });

  it("shows at most five entries followed by the number remaining", () => {
    const imminent = Array.from({ length: 7 }, (_value, index) =>
      makeInfo({ model_name: `model-${index + 1}` }),
    );
    mockedUseModelDeprecations.mockReturnValue(queryResult(makeResponse({ imminent })));

    render(<ModelDeprecationBanner />);

    const entries = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(entries).toHaveLength(6);
    expect(entries[4]).toHaveTextContent("model-5 deprecates");
    expect(entries[5]).toHaveTextContent("and 2 more");
    expect(screen.queryByText(/model-6/)).not.toBeInTheDocument();
  });
});
