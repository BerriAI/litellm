import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ToolPermissionRulesEditor, {
  ToolPermissionConfig,
} from "./ToolPermissionRulesEditor";

describe("ToolPermissionRulesEditor", () => {
  it("renders empty state and lets users add a new rule", async () => {
    const onChange = vi.fn();
    render(<ToolPermissionRulesEditor value={undefined} onChange={onChange} />);

    expect(screen.getByText(/No tool rules added yet/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /add rule/i }));

    expect(onChange).toHaveBeenCalled();
    const payload = onChange.mock.calls[0][0] as ToolPermissionConfig;
    expect(payload.rules).toHaveLength(1);
    expect(payload.rules[0].decision).toBe("allow");
  });

  it("captures violation message and argument constraints", async () => {
    let latestConfig: ToolPermissionConfig | null = null;
    const initialConfig: ToolPermissionConfig = {
      rules: [
        {
          id: "allow_bash",
          tool_name: "Bash",
          decision: "allow",
        },
      ],
      default_action: "deny",
      on_disallowed_action: "block",
      violation_message_template: "",
    };

    const Wrapper = () => {
      const [state, setState] = React.useState(initialConfig);
      const handleChange = (next: ToolPermissionConfig) => {
        latestConfig = next;
        setState(next);
      };
      return <ToolPermissionRulesEditor value={state} onChange={handleChange} />;
    };

    render(<Wrapper />);

    await userEvent.click(screen.getByRole("button", { name: /restrict tool arguments/i }));
    const initialInput = await screen.findByPlaceholderText(/messages\[0\].content/i);
    await userEvent.clear(initialInput);
    fireEvent.change(initialInput, { target: { value: "input.location" } });

    const violationArea = await screen.findByPlaceholderText(/violates our org policy/i);
    await userEvent.clear(violationArea);
    fireEvent.change(violationArea, { target: { value: "Do not run bash" } });

    await waitFor(() => {
      expect(latestConfig).not.toBeNull();
      expect(latestConfig?.rules[0].allowed_param_patterns).toEqual({ "input.location": "" });
      expect(latestConfig?.violation_message_template).toBe("Do not run bash");
    });
  });
});
