import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import RouterConfigBuilder from "./RouterConfigBuilder";

const MOCK_MODEL_INFO = [
  { model_group: "gpt-4", mode: "chat" },
  { model_group: "gpt-3.5-turbo", mode: "chat" },
  { model_group: "claude-3-opus", mode: "chat" },
];

describe("RouterConfigBuilder", () => {
  it("should render", () => {
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} />);

    expect(screen.getByText("Routes Configuration")).toBeInTheDocument();
  });

  it("should display Add Route button", () => {
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} />);

    expect(screen.getByRole("button", { name: /add route/i })).toBeInTheDocument();
  });

  it("should show empty state when no routes are configured", () => {
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} />);

    expect(screen.getByText(/no routes configured/i)).toBeInTheDocument();
  });

  it("should add a route when Add Route is clicked", async () => {
    const user = userEvent.setup();
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} />);

    await user.click(screen.getByRole("button", { name: /add route/i }));

    expect(screen.getByText("Route 1: Unnamed")).toBeInTheDocument();
  });

  it("should call onChange when a route is added", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /add route/i }));

    expect(onChange).toHaveBeenCalledWith({
      routes: [
        expect.objectContaining({
          name: "",
          utterances: [],
          description: "",
          score_threshold: 0.5,
        }),
      ],
    });
  });

  it("should initialize routes from value prop", async () => {
    const value = {
      routes: [
        {
          name: "gpt-4",
          utterances: ["hello", "hi"],
          description: "For greetings",
          score_threshold: 0.7,
        },
      ],
    };
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-4")).toBeInTheDocument();
    });
  });

  it("should support both name and model fields in value prop", async () => {
    const value = {
      routes: [{ model: "gpt-3.5-turbo", utterances: [], description: "", score_threshold: 0.5 }],
    };
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-3.5-turbo")).toBeInTheDocument();
    });
  });

  it("should remove a route when delete button is clicked", async () => {
    const user = userEvent.setup();
    const value = {
      routes: [
        {
          name: "gpt-4",
          utterances: [],
          description: "",
          score_threshold: 0.5,
        },
      ],
    };
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-4")).toBeInTheDocument();
    });

    const deleteButton = screen.getByRole("button", { name: "delete" });
    await user.click(deleteButton);

    await waitFor(() => {
      expect(screen.queryByText("Route 1: gpt-4")).not.toBeInTheDocument();
      expect(screen.getByText(/no routes configured/i)).toBeInTheDocument();
    });
  });

  it("should call onChange when route is removed", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const value = {
      routes: [
        {
          name: "gpt-4",
          utterances: [],
          description: "",
          score_threshold: 0.5,
        },
      ],
    };
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} onChange={onChange} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-4")).toBeInTheDocument();
    });

    const deleteButton = screen.getByRole("button", { name: "delete" });
    await user.click(deleteButton);

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({ routes: [] });
    });
  });




  it("should update route when description is changed", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const value = {
      routes: [
        {
          name: "gpt-4",
          utterances: [],
          description: "",
          score_threshold: 0.5,
        },
      ],
    };
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} onChange={onChange} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-4")).toBeInTheDocument();
    });

    const descriptionInput = screen.getByPlaceholderText("Describe when this route should be used...");
    await user.type(descriptionInput, "For code generation");

    await waitFor(() => {
      const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1];
      expect(lastCall[0].routes[0].description).toBe("For code generation");
    });
  });

  it("should update route when score threshold is changed", async () => {
    const onChange = vi.fn();
    const value = {
      routes: [
        {
          name: "gpt-4",
          utterances: [],
          description: "",
          score_threshold: 0.5,
        },
      ],
    };
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} onChange={onChange} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-4")).toBeInTheDocument();
    });

    const scoreInput = screen.getByRole("spinbutton");
    fireEvent.change(scoreInput, { target: { value: "0.9" } });

    await waitFor(() => {
      const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1];
      expect(lastCall[0].routes[0].score_threshold).toBe(0.9);
    });
  });

  it("should add multiple routes", async () => {
    const user = userEvent.setup();
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} />);

    await user.click(screen.getByRole("button", { name: /add route/i }));
    await user.click(screen.getByRole("button", { name: /add route/i }));

    expect(screen.getByText("Route 1: Unnamed")).toBeInTheDocument();
    expect(screen.getByText("Route 2: Unnamed")).toBeInTheDocument();
  });

  it("should toggle JSON preview visibility", async () => {
    const user = userEvent.setup();
    const { container } = render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} />);

    expect(screen.getByText("JSON Preview")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show" })).toBeInTheDocument();
    expect(container.querySelector("pre")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show" }));

    expect(screen.getByRole("button", { name: "Hide" })).toBeInTheDocument();
    expect(container.querySelector("pre")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Hide" }));

    expect(screen.getByRole("button", { name: "Show" })).toBeInTheDocument();
    expect(container.querySelector("pre")).not.toBeInTheDocument();
  });

  it("should display JSON preview with route data when routes exist", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <RouterConfigBuilder
        modelInfo={MOCK_MODEL_INFO}
        value={{
          routes: [
            { name: "gpt-4", utterances: ["hello"], description: "test", score_threshold: 0.8 },
          ],
        }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-4")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Show" }));

    const preElement = container.querySelector("pre");
    expect(preElement).toBeInTheDocument();
    expect(preElement?.textContent).toContain("gpt-4");
    expect(preElement?.textContent).toContain("hello");
    expect(preElement?.textContent).toContain("0.8");
  });

  it("should display model selector with options from modelInfo", async () => {
    const value = {
      routes: [
        { name: "", utterances: [], description: "", score_threshold: 0.5 },
      ],
    };
    render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: Unnamed")).toBeInTheDocument();
    });

    expect(screen.getByText("Model")).toBeInTheDocument();
    const comboboxes = screen.getAllByRole("combobox");
    expect(comboboxes.length).toBeGreaterThan(0);
  });

  it("should clear routes when value prop changes to empty", async () => {
    const value = {
      routes: [
        {
          name: "gpt-4",
          utterances: [],
          description: "",
          score_threshold: 0.5,
        },
      ],
    };
    const { rerender } = render(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={value} />);

    await waitFor(() => {
      expect(screen.getByText("Route 1: gpt-4")).toBeInTheDocument();
    });

    rerender(<RouterConfigBuilder modelInfo={MOCK_MODEL_INFO} value={{ routes: [] }} />);

    await waitFor(() => {
      expect(screen.getByText(/no routes configured/i)).toBeInTheDocument();
    });
  });
});
