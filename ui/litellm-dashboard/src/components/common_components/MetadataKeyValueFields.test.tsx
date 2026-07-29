import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Form } from "antd";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { TeamMetadataField } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import MetadataKeyValueFields, {
  MetadataPair,
  metadataObjectToPairs,
  metadataPairsToObject,
  schemaMetadataToObject,
} from "./MetadataKeyValueFields";

describe("metadataObjectToPairs", () => {
  it("returns an empty list for null or undefined metadata", () => {
    expect(metadataObjectToPairs(null)).toEqual([]);
    expect(metadataObjectToPairs(undefined)).toEqual([]);
  });

  it("keeps plain string values as-is", () => {
    expect(metadataObjectToPairs({ department: "research" })).toEqual([{ key: "department", value: "research" }]);
  });

  it("serializes non-string values as JSON", () => {
    expect(
      metadataObjectToPairs({
        tier: 3,
        beta: true,
        config: { region: "us" },
        tags: ["a", "b"],
        empty: null,
      }),
    ).toEqual([
      { key: "tier", value: "3" },
      { key: "beta", value: "true" },
      { key: "config", value: '{"region":"us"}' },
      { key: "tags", value: '["a","b"]' },
      { key: "empty", value: "null" },
    ]);
  });

  it("quotes string values that would otherwise parse as JSON, so types round-trip", () => {
    expect(metadataObjectToPairs({ code: "42", flag: "true" })).toEqual([
      { key: "code", value: '"42"' },
      { key: "flag", value: '"true"' },
    ]);
  });

  it("filters out excluded keys", () => {
    expect(
      metadataObjectToPairs({ department: "research", logging: [{ callback_name: "langfuse" }] }, new Set(["logging"])),
    ).toEqual([{ key: "department", value: "research" }]);
  });
});

describe("metadataPairsToObject", () => {
  it("returns an empty object for undefined pairs", () => {
    expect(metadataPairsToObject(undefined)).toEqual({});
  });

  it("keeps plain text values as strings", () => {
    expect(metadataPairsToObject([{ key: "department", value: "research" }])).toEqual({ department: "research" });
  });

  it("parses JSON values into their typed form", () => {
    expect(
      metadataPairsToObject([
        { key: "tier", value: "3" },
        { key: "beta", value: "true" },
        { key: "config", value: '{"region":"us"}' },
        { key: "code", value: '"42"' },
      ]),
    ).toEqual({ tier: 3, beta: true, config: { region: "us" }, code: "42" });
  });

  it("skips rows without a key and defaults a missing value to an empty string", () => {
    expect(metadataPairsToObject([{ key: "", value: "orphan" }, undefined, { key: "kept" }])).toEqual({ kept: "" });
  });

  it("round-trips a mixed-type metadata object losslessly", () => {
    const metadata = {
      department: "research",
      code: "42",
      tier: 3,
      beta: true,
      config: { region: "us", replicas: 2 },
    };
    expect(metadataPairsToObject(metadataObjectToPairs(metadata))).toEqual(metadata);
  });
});

describe("schemaMetadataToObject", () => {
  it("should return an empty object for null or undefined values", () => {
    expect(schemaMetadataToObject(null)).toEqual({});
    expect(schemaMetadataToObject(undefined)).toEqual({});
  });

  it("should omit blank and whitespace-only values", () => {
    expect(schemaMetadataToObject({ cost_center: "CC-1001", app_name: "", notes: "   ", missing: undefined })).toEqual({
      cost_center: "CC-1001",
    });
  });

  it("should parse JSON values into their typed form", () => {
    expect(schemaMetadataToObject({ tier: "3", beta: "true", code: '"42"' })).toEqual({
      tier: 3,
      beta: true,
      code: "42",
    });
  });
});

interface HarnessProps {
  onFinish: (values: { metadata?: MetadataPair[]; schema_metadata?: Record<string, string | undefined> }) => void;
  initialMetadata?: MetadataPair[];
  schemaFields?: TeamMetadataField[];
  schemaLoading?: boolean;
  sourceMetadata?: Record<string, unknown>;
}

const Harness: React.FC<HarnessProps> = ({
  onFinish,
  initialMetadata,
  schemaFields,
  schemaLoading,
  sourceMetadata,
}) => {
  const [form] = Form.useForm();
  return (
    <Form form={form} onFinish={onFinish} initialValues={{ metadata: initialMetadata }}>
      <MetadataKeyValueFields
        form={form}
        schemaFields={schemaFields}
        schemaLoading={schemaLoading}
        sourceMetadata={sourceMetadata}
      />
      <button type="submit">Save</button>
    </Form>
  );
};

describe("MetadataKeyValueFields", () => {
  it("renders one row per existing pair", () => {
    render(
      <Harness
        onFinish={vi.fn()}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "tier", value: "3" },
        ]}
      />,
    );

    const keyInputs = screen.getAllByPlaceholderText("Key");
    const valueInputs = screen.getAllByPlaceholderText("Value");
    expect(keyInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["department", "tier"]);
    expect(valueInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["research", "3"]);
  });

  it("adds a row and submits the entered pair", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} />);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await user.type(screen.getByPlaceholderText("Key"), "cost_center");
    await user.type(screen.getByPlaceholderText("Value"), "eng-1");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "cost_center", value: "eng-1" }] });
    });
  });

  it("removes a row when its remove icon is clicked", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(
      <Harness
        onFinish={onFinish}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "tier", value: "3" },
        ]}
      />,
    );

    await user.click(screen.getAllByLabelText("Remove key-value pair")[0]);
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "tier", value: "3" }] });
    });
  });

  it("blocks submission on duplicate keys", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(
      <Harness
        onFinish={onFinish}
        initialMetadata={[
          { key: "department", value: "research" },
          { key: "department", value: "sales" },
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getAllByText("Duplicate key").length).toBeGreaterThan(0);
    });
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("blocks submission when a row is missing its key", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} />);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await user.type(screen.getByPlaceholderText("Value"), "orphan");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByText("Missing key")).toBeInTheDocument();
    });
    expect(onFinish).not.toHaveBeenCalled();
  });
});

describe("MetadataKeyValueFields with a declared schema", () => {
  const schema: TeamMetadataField[] = [
    { key: "cost_center", label: "Cost Center", required: true, description: "Cost center code" },
    { key: "app_name", label: "Application Name" },
  ];

  it("should render a locked pair row per declared field with its description and no remove icon", () => {
    render(<Harness onFinish={vi.fn()} schemaFields={schema} initialMetadata={[{ key: "notes", value: "x" }]} />);

    const costCenterKeyInput = screen.getByDisplayValue("cost_center");
    expect(costCenterKeyInput).toBeDisabled();
    expect(screen.getByDisplayValue("app_name")).toBeDisabled();
    expect(screen.getByPlaceholderText("Cost Center")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Application Name")).toBeInTheDocument();
    expect(screen.getByText("Cost center code")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Remove key-value pair")).toHaveLength(1);
  });

  it("should mark only required declared fields with an asterisk", () => {
    render(<Harness onFinish={vi.fn()} schemaFields={schema} />);

    expect(screen.getAllByTitle("Required")).toHaveLength(1);
  });

  it("should submit declared field values under schema_metadata", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} schemaFields={schema} />);

    await user.type(screen.getByPlaceholderText("Cost Center"), "CC-1001");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith(
        expect.objectContaining({ schema_metadata: expect.objectContaining({ cost_center: "CC-1001" }) }),
      );
    });
  });

  it("should block submission when a required declared field is blank", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} schemaFields={schema} />);

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByText("Cost Center is required")).toBeInTheDocument();
    });
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("should prefill declared fields from sourceMetadata and prune them from free-form rows", async () => {
    render(
      <Harness
        onFinish={vi.fn()}
        schemaFields={schema}
        initialMetadata={[
          { key: "cost_center", value: "CC-1001" },
          { key: "notes", value: "x" },
        ]}
        sourceMetadata={{ cost_center: "CC-1001", notes: "x" }}
      />,
    );

    expect(screen.getByPlaceholderText("Cost Center")).toHaveValue("CC-1001");
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText("Key").map((input) => (input as HTMLInputElement).value)).toEqual([
        "notes",
      ]);
    });
  });

  it("should reject a free-form key that matches a declared field", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} schemaFields={schema} />);

    await user.type(screen.getByPlaceholderText("Cost Center"), "CC-1001");
    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await user.type(screen.getByPlaceholderText("Key"), "cost_center");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByText("Key is managed by the fields above")).toBeInTheDocument();
    });
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("should show a skeleton instead of the editor while the schema is loading", () => {
    render(<Harness onFinish={vi.fn()} schemaLoading />);

    expect(screen.getByTestId("metadata-schema-skeleton")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /add key-value pair/i })).not.toBeInTheDocument();
  });
});
