import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import MoyaiConnected from "./MoyaiConnected";

vi.mock("@/utils/uiHref", () => ({ uiHref: (seg: string) => `/ui/${seg}` }));

describe("MoyaiConnected", () => {
  it("shows workspace, key alias, and model count rows with an Open Moyai CTA", () => {
    render(<MoyaiConnected moyaiUrl="https://moyai.example.com" keyAlias="moyai-moyai.example.com" models={42} />);

    expect(screen.getByText("Workspace linked")).toBeInTheDocument();
    expect(screen.getByText("https://moyai.example.com")).toBeInTheDocument();
    expect(screen.getByText("Virtual key issued")).toBeInTheDocument();
    expect(screen.getByText("moyai-moyai.example.com")).toBeInTheDocument();
    expect(screen.getByText(/models available through LiteLLM/)).toHaveTextContent(
      "42 models available through LiteLLM",
    );
    expect(screen.getByRole("link", { name: /Open Moyai/ })).toHaveAttribute("href", "https://moyai.example.com");
    expect(screen.getByRole("link", { name: /Back to AI Gateway/ })).toHaveAttribute("href", "/ui/");
  });

  it("hides the models row when the count is missing", () => {
    render(<MoyaiConnected moyaiUrl="https://moyai.example.com" keyAlias={null} models={null} />);

    expect(screen.queryByText(/models available through LiteLLM/)).not.toBeInTheDocument();
    expect(screen.queryByText("Virtual key issued")).not.toBeInTheDocument();
  });
});
