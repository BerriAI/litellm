import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LlmBadge, McpBadge, AgentBadge } from "./TypeBadges";

describe("TypeBadges", () => {
  describe("LlmBadge", () => {
    it("should render with default 'LLM' text when no count is provided", () => {
      render(<LlmBadge />);
      expect(screen.getByText("LLM")).toBeInTheDocument();
    });

    it("should render the count when provided", () => {
      render(<LlmBadge count={5} />);
      expect(screen.getByText("5")).toBeInTheDocument();
    });

    it("should render count of 0 instead of default text", () => {
      render(<LlmBadge count={0} />);
      expect(screen.getByText("0")).toBeInTheDocument();
    });
  });

  describe("McpBadge", () => {
    it("should render with default 'MCP' text when no count is provided", () => {
      render(<McpBadge />);
      expect(screen.getByText("MCP")).toBeInTheDocument();
    });

    it("should render the count when provided", () => {
      render(<McpBadge count={3} />);
      expect(screen.getByText("3")).toBeInTheDocument();
    });
  });

  describe("AgentBadge", () => {
    it("should render with default 'Agent' text when no count is provided", () => {
      render(<AgentBadge />);
      expect(screen.getByText("Agent")).toBeInTheDocument();
    });

    it("should render the count when provided", () => {
      render(<AgentBadge count={12} />);
      expect(screen.getByText("12")).toBeInTheDocument();
    });
  });
});
