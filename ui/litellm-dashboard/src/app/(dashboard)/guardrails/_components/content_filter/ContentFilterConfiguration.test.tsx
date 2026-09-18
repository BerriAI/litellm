import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders, screen } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import ContentFilterConfiguration from "./ContentFilterConfiguration";

vi.mock("@/components/networking", () => ({
  validateBlockedWordsFile: vi.fn(),
  getCategoryYaml: vi.fn(),
}));

const PREBUILT = [
  { name: "us_ssn", display_name: "US Social Security Number", category: "PII Patterns", description: "d" },
];

describe("ContentFilterConfiguration", () => {
  const handlers = {
    onPatternAdd: vi.fn(),
    onPatternRemove: vi.fn(),
    onPatternActionChange: vi.fn(),
    onBlockedWordAdd: vi.fn(),
    onBlockedWordRemove: vi.fn(),
    onBlockedWordUpdate: vi.fn(),
  };

  const renderConfig = (overrides = {}) =>
    renderWithProviders(
      <ContentFilterConfiguration
        prebuiltPatterns={PREBUILT}
        categories={["PII Patterns"]}
        selectedPatterns={[]}
        blockedWords={[]}
        accessToken="test-token"
        {...handlers}
        {...overrides}
      />,
    );

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the pattern and keyword sections", () => {
    renderConfig();

    expect(screen.getByText("Pattern Detection")).toBeInTheDocument();
    expect(
      screen.getByText("Detect sensitive information using regex patterns (SSN, credit cards, API keys, etc.)"),
    ).toBeInTheDocument();
    expect(screen.getByText("Blocked Keywords")).toBeInTheDocument();
    expect(screen.getByText("Block or mask specific sensitive terms and phrases")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /add prebuilt pattern/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /add custom regex/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /add keyword/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /upload yaml file/i })).toBeInTheDocument();
  });

  it("should show the empty states for patterns and keywords", () => {
    renderConfig();

    expect(screen.getByText("No patterns added.")).toBeInTheDocument();
    expect(screen.getByText("No keywords added.")).toBeInTheDocument();
  });

  it("should open the prebuilt pattern modal", async () => {
    const user = userEvent.setup();
    renderConfig();

    expect(screen.queryByText("Pattern type")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add prebuilt pattern/i }));

    expect(await screen.findByText("Pattern type")).toBeInTheDocument();
  });

  it("should open the custom regex modal", async () => {
    const user = userEvent.setup();
    renderConfig();

    expect(screen.queryByText("Add custom regex pattern")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add custom regex/i }));

    expect(await screen.findByText("Add custom regex pattern")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., ID-[0-9]{6}")).toBeInTheDocument();
  });

  it("should open the keyword modal", async () => {
    const user = userEvent.setup();
    renderConfig();

    expect(screen.queryByText("Add blocked keyword")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add keyword/i }));

    expect(await screen.findByText("Add blocked keyword")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter sensitive keyword or phrase")).toBeInTheDocument();
  });

  it("should list already selected patterns and keywords", () => {
    renderConfig({
      selectedPatterns: [
        {
          id: "pattern-1",
          type: "prebuilt" as const,
          name: "us_ssn",
          display_name: "US Social Security Number",
          action: "BLOCK" as const,
        },
      ],
      blockedWords: [{ id: "word-1", keyword: "secret", action: "MASK" as const, description: "Sensitive" }],
    });

    expect(screen.getByText("US Social Security Number")).toBeInTheDocument();
    expect(screen.getByText("secret")).toBeInTheDocument();
    expect(screen.queryByText("No patterns added.")).not.toBeInTheDocument();
    expect(screen.queryByText("No keywords added.")).not.toBeInTheDocument();
  });

  it("should show only the keyword section when the keywords step is requested", () => {
    renderConfig({ showStep: "keywords" });

    expect(screen.getByText("Blocked Keywords")).toBeInTheDocument();
    expect(screen.queryByText("Pattern Detection")).not.toBeInTheDocument();
  });

  it("should show only the pattern section when the patterns step is requested", () => {
    renderConfig({ showStep: "patterns" });

    expect(screen.getByText("Pattern Detection")).toBeInTheDocument();
    expect(screen.queryByText("Blocked Keywords")).not.toBeInTheDocument();
  });
});
