import React from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  FileText as FileTextOutlined,
  Plus as PlusOutlined,
  Trash2 as DeleteOutlined,
} from "lucide-react";
import { getCategoryYaml } from "../../networking";

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
  const [localSelectedCategoryName, setLocalSelectedCategoryName] =
    React.useState<string>("");
  const selectedCategoryName =
    pendingSelection !== undefined ? pendingSelection : localSelectedCategoryName;
  const setSelectedCategoryName =
    onPendingSelectionChange || setLocalSelectedCategoryName;
  const [categoryYaml, setCategoryYaml] = React.useState<{
    [key: string]: string;
  }>({});
  const [categoryFileTypes, setCategoryFileTypes] = React.useState<{
    [key: string]: string;
  }>({});
  const [loadingYaml, setLoadingYaml] = React.useState<{
    [key: string]: boolean;
  }>({});
  const [expandedYamlCategories, setExpandedYamlCategories] = React.useState<
    string[]
  >([]);
  const [previewYaml, setPreviewYaml] = React.useState<string>("");
  const [loadingPreviewYaml, setLoadingPreviewYaml] =
    React.useState<boolean>(false);

  const handleAddCategory = () => {
    if (!selectedCategoryName) {
      return;
    }

    const category = availableCategories.find(
      (c) => c.name === selectedCategoryName,
    );
    if (!category) {
      return;
    }

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
    setPreviewYaml("");
  };

  const fetchCategoryYaml = async (categoryName: string) => {
    if (!accessToken) {
      return;
    }

    if (categoryYaml[categoryName]) {
      return;
    }

    setLoadingYaml((prev) => ({ ...prev, [categoryName]: true }));
    try {
      const data = await getCategoryYaml(accessToken, categoryName);
      let content = data.yaml_content;

      if (data.file_type === "json") {
        try {
          const parsed = JSON.parse(content);
          content = JSON.stringify(parsed, null, 2);
        } catch (e) {
          console.warn(`Failed to format JSON for ${categoryName}:`, e);
        }
      }

      setCategoryYaml((prev) => ({ ...prev, [categoryName]: content }));
      setCategoryFileTypes((prev) => ({
        ...prev,
        [categoryName]: data.file_type || "yaml",
      }));
    } catch (error) {
      console.error(
        `Failed to fetch content for category ${categoryName}:`,
        error,
      );
    } finally {
      setLoadingYaml((prev) => ({ ...prev, [categoryName]: false }));
    }
  };

  React.useEffect(() => {
    if (selectedCategoryName && accessToken) {
      const cachedContent = categoryYaml[selectedCategoryName];
      if (cachedContent) {
        setPreviewYaml(cachedContent);
        return;
      }

      setLoadingPreviewYaml(true);
      getCategoryYaml(accessToken, selectedCategoryName)
        .then((data) => {
          let content = data.yaml_content;

          if (data.file_type === "json") {
            try {
              const parsed = JSON.parse(content);
              content = JSON.stringify(parsed, null, 2);
            } catch (e) {
              console.warn(
                `Failed to format JSON for ${selectedCategoryName}:`,
                e,
              );
            }
          }

          setPreviewYaml(content);
          setCategoryYaml((prev) => ({
            ...prev,
            [selectedCategoryName]: content,
          }));
          setCategoryFileTypes((prev) => ({
            ...prev,
            [selectedCategoryName]: data.file_type || "yaml",
          }));
        })
        .catch((error) => {
          console.error(
            `Failed to fetch preview content for category ${selectedCategoryName}:`,
            error,
          );
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

  const unselectedCategories = availableCategories.filter(
    (cat) => !selectedCategories.some((sel) => sel.category === cat.name),
  );

  return (
    <Card className="p-4 space-y-4">
      <div className="flex justify-between items-center flex-wrap gap-2">
        <h5 className="text-base font-semibold m-0">Blocked topics</h5>
        <span className="text-xs text-muted-foreground font-normal">
          Select topics to block using keyword and semantic analysis
        </span>
      </div>

      <div className="flex gap-2">
        <div className="flex-1">
          <Select
            value={selectedCategoryName || ""}
            onValueChange={(v) => setSelectedCategoryName(v)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a content category" />
            </SelectTrigger>
            <SelectContent>
              {unselectedCategories.length === 0 ? (
                <div className="py-2 px-3 text-sm text-muted-foreground">
                  No categories available
                </div>
              ) : (
                unselectedCategories.map((cat) => (
                  <SelectItem key={cat.name} value={cat.name}>
                    <div>
                      <div className="font-medium">{cat.display_name}</div>
                      <div className="text-xs text-muted-foreground mt-0.5">
                        {cat.description}
                      </div>
                    </div>
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
        </div>
        <Button
          onClick={handleAddCategory}
          disabled={!selectedCategoryName}
        >
          <PlusOutlined className="h-4 w-4 mr-1" />
          Add
        </Button>
      </div>

      {selectedCategoryName && (
        <div className="p-3 bg-muted border border-border rounded">
          <div className="mb-2 text-sm font-medium">
            Preview:{" "}
            {
              availableCategories.find((c) => c.name === selectedCategoryName)
                ?.display_name
            }
            {categoryFileTypes[selectedCategoryName] && (
              <span className="ml-2 text-xs text-muted-foreground font-normal">
                ({categoryFileTypes[selectedCategoryName]?.toUpperCase()})
              </span>
            )}
          </div>
          {loadingPreviewYaml ? (
            <div className="p-4 text-center text-muted-foreground">
              Loading content...
            </div>
          ) : previewYaml ? (
            <pre className="bg-background p-3 rounded overflow-auto max-h-[300px] max-w-full text-xs leading-relaxed m-0 border border-border whitespace-pre-wrap break-words">
              <code>{previewYaml}</code>
            </pre>
          ) : (
            <div className="p-2 text-center text-muted-foreground text-xs">
              Unable to load category content
            </div>
          )}
        </div>
      )}

      {selectedCategories.length > 0 ? (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Category</TableHead>
                <TableHead className="w-[150px]">Action</TableHead>
                <TableHead className="w-[180px]">Severity Threshold</TableHead>
                <TableHead className="w-[120px]" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {selectedCategories.map((record) => {
                const category = availableCategories.find(
                  (c) => c.name === record.category,
                );
                return (
                  <TableRow key={record.id}>
                    <TableCell>
                      <div>
                        <div className="font-medium">{record.display_name}</div>
                        {category?.description && (
                          <div className="text-xs text-muted-foreground mt-1">
                            {category.description}
                          </div>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Select
                        value={record.action}
                        onValueChange={(value) =>
                          onCategoryUpdate(record.id, "action", value)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="BLOCK">
                            <Badge variant="destructive">BLOCK</Badge>
                          </SelectItem>
                          <SelectItem value="MASK">
                            <Badge
                              variant="secondary"
                              className="bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300"
                            >
                              MASK
                            </Badge>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </TableCell>
                    <TableCell>
                      <Select
                        value={record.severity_threshold}
                        onValueChange={(value) =>
                          onCategoryUpdate(
                            record.id,
                            "severity_threshold",
                            value,
                          )
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="medium">Medium</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                        </SelectContent>
                      </Select>
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onCategoryRemove(record.id)}
                      >
                        <DeleteOutlined className="h-4 w-4 mr-1" />
                        Remove
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>

          <Accordion
            type="multiple"
            value={expandedYamlCategories}
            onValueChange={(values) => {
              const oldExpanded = new Set(expandedYamlCategories);
              values.forEach((key) => {
                if (!oldExpanded.has(key) && !categoryYaml[key]) {
                  fetchCategoryYaml(key);
                }
              });
              setExpandedYamlCategories(values);
            }}
          >
            {selectedCategories.map((category) => {
              const fileType = categoryFileTypes[category.category] || "yaml";
              const fileTypeLabel = fileType.toUpperCase();
              return (
                <AccordionItem
                  key={category.category}
                  value={category.category}
                >
                  <AccordionTrigger>
                    <div className="flex items-center gap-2">
                      <FileTextOutlined className="h-4 w-4" />
                      <span>
                        View {fileTypeLabel} for {category.display_name}
                      </span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    {loadingYaml[category.category] ? (
                      <div className="p-4 text-center text-muted-foreground">
                        Loading content...
                      </div>
                    ) : categoryYaml[category.category] ? (
                      <pre className="bg-muted p-4 rounded overflow-auto max-h-[400px] text-xs leading-relaxed m-0">
                        <code>{categoryYaml[category.category]}</code>
                      </pre>
                    ) : (
                      <div className="p-4 text-center text-muted-foreground">
                        Content will load when expanded
                      </div>
                    )}
                  </AccordionContent>
                </AccordionItem>
              );
            })}
          </Accordion>
        </>
      ) : (
        <div className="text-center py-6 text-muted-foreground border border-dashed border-border rounded">
          No blocked topics selected. Add topics to detect and block harmful
          content.
        </div>
      )}
    </Card>
  );
};

export default ContentCategoryConfiguration;
