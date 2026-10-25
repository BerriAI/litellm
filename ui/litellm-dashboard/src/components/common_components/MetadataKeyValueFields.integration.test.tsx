import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { TeamMetadataField } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import { useZodForm } from "@/lib/forms/useZodForm";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import MetadataKeyValueFields, {
  MetadataPair,
  metadataObjectToPairs,
  metadataPairsSchema,
  metadataPairsToObject,
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

  it("drops an empty key, which the form can never submit, so an API-written one does not block saving", () => {
    expect(metadataObjectToPairs({ "": { displayName: "okta-push-group" }, scim_managed: true })).toEqual([
      { key: "scim_managed", value: "true" },
    ]);
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

interface HarnessProps {
  onFinish: (values: { metadata?: MetadataPair[] }) => void;
  initialMetadata?: MetadataPair[];
  schemaFields?: TeamMetadataField[];
  schemaLoading?: boolean;
  fieldsHidden?: boolean;
  serverError?: string;
}

const harnessSchema = z.object({ metadata: metadataPairsSchema });

const Harness: React.FC<HarnessProps> = ({
  onFinish,
  initialMetadata,
  schemaFields,
  schemaLoading,
  fieldsHidden,
  serverError,
}) => {
  const form = useZodForm(harnessSchema, { defaultValues: { metadata: initialMetadata ?? [] } });
  React.useEffect(() => {
    if (serverError) form.setError("metadata.0.key", { type: "server", message: serverError });
  }, [form, serverError]);
  return (
    <form onSubmit={form.handleSubmit((values) => onFinish(values))}>
      {!fieldsHidden && (
        <MetadataKeyValueFields
          control={form.control}
          getValues={form.getValues}
          name="metadata"
          schemaFields={schemaFields}
          schemaLoading={schemaLoading}
        />
      )}
      <button type="submit">Save</button>
    </form>
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

    const keyInputs = screen.getAllByPlaceholderText("Metadata key");
    const valueInputs = screen.getAllByPlaceholderText("Value");
    expect(keyInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["department", "tier"]);
    expect(valueInputs.map((input) => (input as HTMLInputElement).value)).toEqual(["research", "3"]);
  });

  it("adds a row and submits the entered pair", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} />);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    fireEvent.change(screen.getByPlaceholderText("Metadata key"), { target: { value: "cost_center" } });
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "eng-1" } });
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
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "orphan" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByText("Missing key")).toBeInTheDocument();
    });
    expect(onFinish).not.toHaveBeenCalled();
  });
});

describe("MetadataKeyValueFields 校验语言", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it.each([
    ["missing", "Missing key", "请输入键", 1],
    ["duplicate", "Duplicate key", "键重复", 2],
  ] as const)("已有本地错误随语言双向切换，类型=%s", async (kind, english, chinese, count) => {
    const user = userEvent.setup({ delay: null });
    const onFinish = vi.fn();
    const initialMetadata =
      kind === "missing"
        ? [{ key: "", value: "Home" }]
        : [
            { key: "Home", value: "Home" },
            { key: "Home", value: '{"Home":3}' },
          ];
    render(
      <LanguageProvider defaultLanguage="en" userId={null} storageScope={`metadata-validation-${kind}`}>
        <LanguageSelector />
        <Harness onFinish={onFinish} initialMetadata={initialMetadata} />
      </LanguageProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findAllByText(english)).toHaveLength(count);
    await user.selectOptions(screen.getByLabelText("Interface language"), "zh-CN");
    expect(await screen.findAllByText(chinese)).toHaveLength(count);
    expect(screen.queryByText(english)).not.toBeInTheDocument();
    expect(screen.getAllByPlaceholderText("值")[0]).toHaveValue("Home");
    expect(onFinish).not.toHaveBeenCalled();
    await user.selectOptions(screen.getByLabelText("界面语言"), "en");
    expect(await screen.findAllByText(english)).toHaveLength(count);
    expect(screen.queryByText(chinese)).not.toBeInTheDocument();
    const keys = screen.getAllByPlaceholderText("Metadata key");
    expect(keys[0]).toHaveValue(kind === "missing" ? "" : "Home");
    fireEvent.change(keys[kind === "missing" ? 0 : 1], { target: { value: "HomeDraft" } });
    await user.click(screen.getByRole("button", { name: "Save" }));
    const expectedMetadata =
      kind === "missing"
        ? [{ key: "HomeDraft", value: "Home" }]
        : [
            { key: "Home", value: "Home" },
            { key: "HomeDraft", value: '{"Home":3}' },
          ];
    await waitFor(() => expect(onFinish).toHaveBeenCalledWith({ metadata: expectedMetadata }));
  });

  it("服务端同名错误保持原文，不当作本地校验翻译", async () => {
    const user = userEvent.setup({ delay: null });
    const onFinish = vi.fn();
    render(
      <LanguageProvider defaultLanguage="en" userId={null} storageScope="metadata-server-error">
        <LanguageSelector />
        <Harness onFinish={onFinish} initialMetadata={[{ key: "Home", value: "Home" }]} serverError="Missing key" />
      </LanguageProvider>,
    );
    expect(await screen.findByText("Missing key")).toBeVisible();
    await user.selectOptions(screen.getByLabelText("Interface language"), "zh-CN");
    expect(screen.getByText("Missing key")).toBeVisible();
    expect(screen.queryByText("请输入键")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("键名")).toHaveValue("Home");
    expect(onFinish).not.toHaveBeenCalled();
  });
});

describe("MetadataKeyValueFields with a declared schema", () => {
  const schema: TeamMetadataField[] = [
    { key: "cost_center", label: "Cost Center" },
    { key: "app_name", label: "Application Name" },
  ];

  it("should render each declared key as a fixed label with only the value editable", async () => {
    render(<Harness onFinish={vi.fn()} schemaFields={schema} />);

    await waitFor(() => {
      expect(screen.getAllByTestId("metadata-schema-label").map((label) => label.textContent)).toEqual([
        "Cost Center",
        "Application Name",
      ]);
    });
    expect(screen.queryByPlaceholderText("Metadata key")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Remove key-value pair")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Cost Center")).toHaveAttribute("placeholder", "Value");
  });

  it("should fall back to the key when a declared field has no label", async () => {
    render(<Harness onFinish={vi.fn()} schemaFields={[{ key: "cost_center" }]} />);

    expect(await screen.findByTestId("metadata-schema-label")).toHaveTextContent("cost_center");
  });

  it("should treat an existing pair that matches a declared key as fixed too", async () => {
    render(
      <Harness
        onFinish={vi.fn()}
        schemaFields={[{ key: "cost_center", label: "Cost Center" }]}
        initialMetadata={[
          { key: "cost_center", value: "CC-1001" },
          { key: "region", value: "us" },
        ]}
      />,
    );

    expect(await screen.findByLabelText("Cost Center")).toHaveValue("CC-1001");
    expect(screen.getAllByPlaceholderText("Metadata key").map((input) => (input as HTMLInputElement).value)).toEqual([
      "region",
    ]);
    expect(screen.getAllByLabelText("Remove key-value pair")).toHaveLength(1);
  });

  it("should submit a prepopulated key with its typed value", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} schemaFields={[{ key: "cost_center", label: "Cost Center" }]} />);

    fireEvent.change(await screen.findByPlaceholderText("Value"), { target: { value: "CC-1001" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({ metadata: [{ key: "cost_center", value: "CC-1001" }] });
    });
  });

  it("should not add a second row for keys already present in the form", async () => {
    render(
      <Harness onFinish={vi.fn()} schemaFields={schema} initialMetadata={[{ key: "cost_center", value: "CC-1001" }]} />,
    );

    await waitFor(() => {
      expect(screen.getAllByTestId("metadata-schema-label").map((label) => label.textContent)).toEqual([
        "Cost Center",
        "Application Name",
      ]);
    });
    expect(screen.getAllByPlaceholderText("Value").map((input) => (input as HTMLInputElement).value)).toEqual([
      "CC-1001",
      "",
    ]);
  });

  it("should still let the user add and remove free-form pairs below the declared keys", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} schemaFields={[{ key: "cost_center", label: "Cost Center" }]} />);

    await screen.findByTestId("metadata-schema-label");
    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    fireEvent.change(screen.getByPlaceholderText("Metadata key"), { target: { value: "region" } });
    fireEvent.change(screen.getAllByPlaceholderText("Value")[1], { target: { value: "us" } });
    fireEvent.change(screen.getByLabelText("Cost Center"), { target: { value: "CC-1001" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith({
        metadata: [
          { key: "cost_center", value: "CC-1001" },
          { key: "region", value: "us" },
        ],
      });
    });

    await user.click(screen.getByLabelText("Remove key-value pair"));
    await waitFor(() => {
      expect(screen.queryByPlaceholderText("Metadata key")).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("metadata-schema-label")).toBeInTheDocument();
  });

  it("should keep duplicate keys in a free-form row", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(<Harness onFinish={onFinish} schemaFields={[{ key: "cost_center", label: "Cost Center" }]} />);

    expect(await screen.findByTestId("metadata-schema-label")).toHaveTextContent("Cost Center");
    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    fireEvent.change(screen.getByPlaceholderText("Metadata key"), { target: { value: "cost_center" } });

    expect(screen.getAllByPlaceholderText("Metadata key")).toHaveLength(1);
    expect(screen.getByPlaceholderText("Metadata key")).toHaveValue("cost_center");
    expect(screen.getAllByLabelText("Remove key-value pair")).toHaveLength(1);
    expect(screen.getAllByTestId("metadata-schema-label")).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(screen.getByText("Duplicate key")).toBeInTheDocument();
    });
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("should keep an existing row above a declared row editable while a declared key is typed into it", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    render(
      <Harness
        onFinish={onFinish}
        initialMetadata={[{ key: "region", value: "us" }]}
        schemaFields={[
          { key: "cost", label: "Cost" },
          { key: "cost_center", label: "Cost Center" },
        ]}
      />,
    );
    await waitFor(() => {
      expect(screen.getAllByTestId("metadata-schema-label").map((label) => label.textContent)).toEqual([
        "Cost",
        "Cost Center",
      ]);
    });

    const keyInput = screen.getByPlaceholderText("Metadata key");
    await user.clear(keyInput);
    await user.type(keyInput, "cost_center");

    expect(keyInput).toHaveValue("cost_center");
    expect(keyInput).toHaveFocus();

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    expect(screen.getAllByPlaceholderText("Metadata key").map((input) => (input as HTMLInputElement).value)).toEqual([
      "cost_center",
      "",
    ]);
    await user.click(screen.getAllByLabelText("Remove key-value pair")[1]);

    expect(screen.getAllByPlaceholderText("Metadata key").map((input) => (input as HTMLInputElement).value)).toEqual([
      "cost_center",
    ]);
    expect(screen.getAllByLabelText("Remove key-value pair")).toHaveLength(1);
    expect(screen.getAllByTestId("metadata-schema-label").map((label) => label.textContent)).toEqual([
      "Cost",
      "Cost Center",
    ]);

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Duplicate key")).toBeInTheDocument();
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("should keep one editable row per duplicated declared key when the editor remounts", async () => {
    const user = userEvent.setup();
    const onFinish = vi.fn();
    const schemaFields = [{ key: "cost_center", label: "Cost Center" }];
    const initialMetadata = [{ key: "region", value: "us" }];
    const { rerender } = render(
      <Harness onFinish={onFinish} initialMetadata={initialMetadata} schemaFields={schemaFields} />,
    );
    expect(await screen.findByTestId("metadata-schema-label")).toHaveTextContent("Cost Center");
    fireEvent.change(screen.getByPlaceholderText("Metadata key"), { target: { value: "cost_center" } });

    rerender(
      <Harness onFinish={onFinish} initialMetadata={initialMetadata} schemaFields={schemaFields} fieldsHidden />,
    );
    rerender(<Harness onFinish={onFinish} initialMetadata={initialMetadata} schemaFields={schemaFields} />);

    expect(screen.getAllByTestId("metadata-schema-label")).toHaveLength(1);
    expect(screen.getByPlaceholderText("Metadata key")).toHaveValue("cost_center");
    expect(screen.getAllByLabelText("Remove key-value pair")).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Duplicate key")).toBeInTheDocument();
    expect(onFinish).not.toHaveBeenCalled();
  });

  it("should label only rows whose current key is declared when the schema arrives after an edit", () => {
    const onFinish = vi.fn();
    const initialMetadata = [{ key: "cost_center", value: "1" }];
    const { rerender } = render(<Harness onFinish={onFinish} initialMetadata={initialMetadata} schemaFields={[]} />);
    fireEvent.change(screen.getByPlaceholderText("Metadata key"), { target: { value: "region" } });

    rerender(
      <Harness
        onFinish={onFinish}
        initialMetadata={initialMetadata}
        schemaFields={[{ key: "cost_center", label: "Cost Center" }]}
      />,
    );

    expect(screen.getByPlaceholderText("Metadata key")).toHaveValue("region");
    expect(screen.getAllByTestId("metadata-schema-label").map((label) => label.textContent)).toEqual(["Cost Center"]);
  });

  it("should show a skeleton instead of the editor while the schema is loading", () => {
    render(<Harness onFinish={vi.fn()} schemaLoading />);

    expect(screen.getByTestId("metadata-schema-skeleton")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /add key-value pair/i })).not.toBeInTheDocument();
  });

  it("should seed rows when the schema arrives after an initial loading state", async () => {
    const onFinish = vi.fn();
    const { rerender } = render(<Harness onFinish={onFinish} schemaLoading />);

    rerender(<Harness onFinish={onFinish} schemaFields={schema} schemaLoading={false} />);

    await waitFor(() => {
      expect(screen.getAllByTestId("metadata-schema-label").map((label) => label.textContent)).toEqual([
        "Cost Center",
        "Application Name",
      ]);
    });
  });
});
