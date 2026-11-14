import { describe, it, expect } from "vitest";
import { renderWithProviders, screen } from "@/../tests/test-utils";
import ContentFilterDisplay from "./ContentFilterDisplay";

const PATTERN = {
  id: "pattern-1",
  type: "prebuilt" as const,
  name: "email",
  display_name: "Email address",
  action: "BLOCK" as const,
};

const KEYWORD = {
  id: "word-1",
  keyword: "secret",
  action: "MASK" as const,
  description: "Sensitive term",
};

const CATEGORY = {
  id: "category-1",
  category: "self_harm",
  display_name: "Self Harm",
  action: "BLOCK" as const,
  severity_threshold: "high" as const,
};

describe("ContentFilterDisplay", () => {
  it("should render nothing when there is no content filter data", () => {
    const { container } = renderWithProviders(<ContentFilterDisplay patterns={[]} blockedWords={[]} categories={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("should render the categories section with a configured count", () => {
    renderWithProviders(<ContentFilterDisplay patterns={[]} blockedWords={[]} categories={[CATEGORY]} />);

    expect(screen.getByText("Content Categories")).toBeInTheDocument();
    expect(screen.getByText("1 categories configured")).toBeInTheDocument();
    expect(screen.getByText("Self Harm")).toBeInTheDocument();
    expect(screen.queryByText("Pattern Detection")).not.toBeInTheDocument();
    expect(screen.queryByText("Blocked Keywords")).not.toBeInTheDocument();
  });

  it("should render the patterns section with a configured count", () => {
    renderWithProviders(<ContentFilterDisplay patterns={[PATTERN]} blockedWords={[]} categories={[]} />);

    expect(screen.getByText("Pattern Detection")).toBeInTheDocument();
    expect(screen.getByText("1 patterns configured")).toBeInTheDocument();
    expect(screen.getByText("Email address")).toBeInTheDocument();
    expect(screen.queryByText("Content Categories")).not.toBeInTheDocument();
  });

  it("should render the keywords section with a configured count", () => {
    renderWithProviders(<ContentFilterDisplay patterns={[]} blockedWords={[KEYWORD]} categories={[]} />);

    expect(screen.getByText("Blocked Keywords")).toBeInTheDocument();
    expect(screen.getByText("1 keywords configured")).toBeInTheDocument();
    expect(screen.getByText("secret")).toBeInTheDocument();
    expect(screen.getByText("Sensitive term")).toBeInTheDocument();
  });

  it("should render every section when all three kinds of data are present", () => {
    renderWithProviders(<ContentFilterDisplay patterns={[PATTERN]} blockedWords={[KEYWORD]} categories={[CATEGORY]} />);

    expect(screen.getByText("Content Categories")).toBeInTheDocument();
    expect(screen.getByText("Pattern Detection")).toBeInTheDocument();
    expect(screen.getByText("Blocked Keywords")).toBeInTheDocument();
  });

  it("should render category severity and action as static text in read-only mode", () => {
    renderWithProviders(
      <ContentFilterDisplay patterns={[PATTERN]} blockedWords={[KEYWORD]} categories={[CATEGORY]} readOnly={true} />,
    );

    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("BLOCK")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /delete/i })).toHaveLength(2);
  });

  it("should render category severity and action as editable controls when not read-only", () => {
    renderWithProviders(
      <ContentFilterDisplay patterns={[PATTERN]} blockedWords={[KEYWORD]} categories={[CATEGORY]} readOnly={false} />,
    );

    expect(screen.queryByText("HIGH")).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /delete/i })).toHaveLength(3);
  });
});
