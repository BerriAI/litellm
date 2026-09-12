import React from "react";
import { ChevronRight, FileText, Plus, Trash2 } from "lucide-react";
import type { ColumnDef } from "@tanstack/react-table";
import { getCategoryYaml } from "@/components/networking";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ACTION_ITEMS, SEVERITY_ITEMS } from "./action_options";

interface ContentCategory {
  name: string;
  display_name: string;
  description: string;
  default_action: string;
}

interface SelectedCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

interface ContentCategoryConfigurationProps {
  availableCategories: ContentCategory[];
  selectedCategories: SelectedCategory[];
  onCategoryAdd: (category: SelectedCategory) => void;
  onCategoryRemove: (id: string) => void;
  onCategoryUpdate: (id: string, field: string, value: any) => void;
  accessToken?: string | null;
  pendingSelection?: string;
  onPendingSelectionChange?: (value: string) => void;
}

const ContentCategoryConfiguration: React.FC<ContentCategoryConfigurationProps> = ({
  availableCategories,
  selectedCategories,
  onCategoryAdd,
  onCategoryRemove,
  onCategoryUpdate,
  accessToken,
  pendingSelection,
  onPendingSelectionChange,
}) => {
  // Use controlled state if parent provides it, otherwise use local state
  const [localSelectedCategoryName, setLocalSelectedCategoryName] = React.useState<string>("");
  const selectedCategoryName = pendingSelection !== undefined ? pendingSelection : localSelectedCategoryName;
  const setSelectedCategoryName = onPendingSelectionChange || setLocalSelectedCategoryName;
  const [categoryYaml, setCategoryYaml] = React.useState<{ [key: string]: string }>({});
  const [categoryFileTypes, setCategoryFileTypes] = React.useState<{ [key: string]: string }>({});
  const [loadingYaml, setLoadingYaml] = React.useState<{ [key: string]: boolean }>({});
  const [expandedYamlCategories, setExpandedYamlCategories] = React.useState<string[]>([]);
  const [previewYaml, setPreviewYaml] = React.useState<string>("");
  const [loadingPreviewYaml, setLoadingPreviewYaml] = React.useState<boolean>(false);

  const handleAddCategory = () => {
    if (!selectedCategoryName) {
      return;
    }

    const category = availableCategories.find((c) => c.name === selectedCategoryName);
    if (!category) {
      return;
    }

    // Check if already added
    if (selectedCategories.some((c) => c.category === selectedCategoryName)) {
      return;
    }

    onCategoryAdd({
      id: `category-${Date.now()}`,
      category: category.name,
      display_name: category.display_name,
      action: category.default_action as "BLOCK" | "MASK",
      severity_threshold: "medium",
    });

    setSelectedCategoryName("");
    setPreviewYaml(""); // Clear preview when category is added
  };

  const fetchCategoryYaml = async (categoryName: string) => {
    if (!accessToken) {
      return; // No access token
    }

    // Check if already loaded
    if (categoryYaml[categoryName]) {
      return;
    }

    setLoadingYaml((prev) => ({ ...prev, [categoryName]: true }));
    try {
      const data = await getCategoryYaml(accessToken, categoryName);
      let content = data.yaml_content;

      // Format JSON content for better readability
      if (data.file_type === "json") {
        try {
          const parsed = JSON.parse(content);
          content = JSON.stringify(parsed, null, 2);
        } catch (e) {
          // If parsing fails, use original content
          console.warn(`Failed to format JSON for ${categoryName}:`, e);
        }
      }

      setCategoryYaml((prev) => ({ ...prev, [categoryName]: content }));
      setCategoryFileTypes((prev) => ({ ...prev, [categoryName]: data.file_type || "yaml" }));
    } catch (error) {
      console.error(`Failed to fetch content for category ${categoryName}:`, error);
    } finally {
      setLoadingYaml((prev) => ({ ...prev, [categoryName]: false }));
    }
  };

  // Fetch preview YAML/JSON when a category is selected in dropdown
  React.useEffect(() => {
    if (selectedCategoryName && accessToken) {
      // Check if we already have this content cached
      const cachedContent = categoryYaml[selectedCategoryName];
      if (cachedContent) {
        setPreviewYaml(cachedContent);
        return;
      }

      // Fetch the content for preview
      setLoadingPreviewYaml(true);
      getCategoryYaml(accessToken, selectedCategoryName)
        .then((data) => {
          let content = data.yaml_content;

          // Format JSON content for better readability
          if (data.file_type === "json") {
            try {
              const parsed = JSON.parse(content);
              content = JSON.stringify(parsed, null, 2);
            } catch (e) {
              console.warn(`Failed to format JSON for ${selectedCategoryName}:`, e);
            }
          }

          setPreviewYaml(content);
          // Also cache it for later use
          setCategoryYaml((prev) => ({ ...prev, [selectedCategoryName]: content }));
          setCategoryFileTypes((prev) => ({ ...prev, [selectedCategoryName]: data.file_type || "yaml" }));
        })
        .catch((error) => {
          console.error(`Failed to fetch preview content for category ${selectedCategoryName}:`, error);
          setPreviewYaml("");
        })
        .finally(() => {
          setLoadingPreviewYaml(false);
        });
    } else {
      setPreviewYaml("");
      setLoadingPreviewYaml(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCategoryName, accessToken]);

  const columns: ColumnDef<SelectedCategory>[] = [
    {
      header: "Category",
      accessorKey: "display_name",
      cell: ({ row }) => {
        const category = availableCategories.find((c) => c.name === row.original.category);
        return (
          <div>
            <div className="font-medium">{row.original.display_name}</div>
            {category?.description && <div className="mt-1 text-xs text-muted-foreground">{category.description}</div>}
          </div>
        );
      },
    },
    {
      header: "Action",
      accessorKey: "action",
      size: 150,
      cell: ({ row }) => (
        <Select
          items={ACTION_ITEMS}
          value={row.original.action}
          onValueChange={(value: string | null) => value && onCategoryUpdate(row.original.id, "action", value)}
        >
          <SelectTrigger size="sm" className="w-full" aria-label="Action">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ACTION_ITEMS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                <Badge variant={item.value === "BLOCK" ? "destructive" : "secondary"}>{item.value}</Badge>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ),
    },
    {
      header: "Severity Threshold",
      accessorKey: "severity_threshold",
      size: 180,
      cell: ({ row }) => (
        <Select
          items={SEVERITY_ITEMS}
          value={row.original.severity_threshold}
          onValueChange={(value: string | null) =>
            value && onCategoryUpdate(row.original.id, "severity_threshold", value)
          }
        >
          <SelectTrigger size="sm" className="w-full" aria-label="Severity Threshold">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SEVERITY_ITEMS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ),
    },
    {
      header: "",
      id: "actions",
      size: 80,
      cell: ({ row }) => (
        <Button variant="outline" size="sm" onClick={() => onCategoryRemove(row.original.id)}>
          <Trash2 />
          Remove
        </Button>
      ),
    },
  ];

  const unselectedCategories = availableCategories.filter(
    (cat) => !selectedCategories.some((sel) => sel.category === cat.name),
  );
  const pendingCategory = availableCategories.find((c) => c.name === selectedCategoryName) ?? null;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle>Blocked topics</CardTitle>
          <p className="text-xs font-normal text-muted-foreground">
            Select topics to block using keyword and semantic analysis
          </p>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex gap-2">
          <Combobox
            items={unselectedCategories}
            value={pendingCategory}
            onValueChange={(category: ContentCategory | null) => setSelectedCategoryName(category?.name ?? "")}
            itemToStringLabel={(category: ContentCategory) => category.display_name}
          >
            <ComboboxInput className="w-full" placeholder="Select a content category" />
            <ComboboxContent>
              <ComboboxEmpty>No matching categories</ComboboxEmpty>
              <ComboboxList>
                {(cat: ContentCategory) => (
                  <ComboboxItem key={cat.name} value={cat}>
                    <div>
                      <div className="font-medium">{cat.display_name}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">{cat.description}</div>
                    </div>
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
          <Button onClick={handleAddCategory} disabled={!selectedCategoryName}>
            <Plus />
            Add
          </Button>
        </div>

        {/* Preview box - shown when category is selected but not yet added */}
        {selectedCategoryName && (
          <div className="mb-4 rounded-md border border-border bg-muted/40 p-3">
            <div className="mb-2 text-sm font-medium">
              Preview: {availableCategories.find((c) => c.name === selectedCategoryName)?.display_name}
              {categoryFileTypes[selectedCategoryName] && (
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  ({categoryFileTypes[selectedCategoryName]?.toUpperCase()})
                </span>
              )}
            </div>
            {loadingPreviewYaml ? (
              <div className="p-4 text-center text-muted-foreground">Loading content...</div>
            ) : previewYaml ? (
              <pre className="m-0 max-h-[300px] max-w-full overflow-auto rounded-md border border-border bg-background p-3 text-xs leading-relaxed break-words whitespace-pre-wrap">
                <code>{previewYaml}</code>
              </pre>
            ) : (
              <div className="p-2 text-center text-xs text-muted-foreground">Unable to load category content</div>
            )}
          </div>
        )}

        {selectedCategories.length > 0 ? (
          <>
            <DataTable data={selectedCategories} columns={columns} getRowId={(row) => row.id} size="compact" />
            <div className="mt-4 space-y-2">
              {selectedCategories.map((category) => {
                const fileType = categoryFileTypes[category.category] || "yaml";
                const isExpanded = expandedYamlCategories.includes(category.category);

                return (
                  <Collapsible
                    key={category.category}
                    open={isExpanded}
                    onOpenChange={(open) => {
                      if (open && !categoryYaml[category.category]) {
                        fetchCategoryYaml(category.category);
                      }
                      setExpandedYamlCategories((prev) =>
                        open ? [...prev, category.category] : prev.filter((name) => name !== category.category),
                      );
                    }}
                  >
                    <CollapsibleTrigger className="flex items-center gap-2 text-sm">
                      <ChevronRight className={`size-4 transition-transform ${isExpanded ? "rotate-90" : ""}`} />
                      <FileText className="size-4" />
                      <span>
                        View {fileType.toUpperCase()} for {category.display_name}
                      </span>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      {loadingYaml[category.category] ? (
                        <div className="p-4 text-center text-muted-foreground">Loading content...</div>
                      ) : categoryYaml[category.category] ? (
                        <pre className="m-0 max-h-[400px] overflow-auto rounded-md bg-muted p-4 text-xs leading-relaxed">
                          <code>{categoryYaml[category.category]}</code>
                        </pre>
                      ) : (
                        <div className="p-4 text-center text-muted-foreground">Content will load when expanded</div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>
                );
              })}
            </div>
          </>
        ) : (
          <div className="rounded-md border border-dashed border-border p-6 text-center text-muted-foreground">
            No blocked topics selected. Add topics to detect and block harmful content.
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ContentCategoryConfiguration;
