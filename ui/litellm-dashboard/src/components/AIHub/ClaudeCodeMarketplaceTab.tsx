import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Search } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import {
  extractCategories,
  filterPluginsByCategory,
  filterPluginsBySearch,
} from "../claude_code_plugins/helpers";
import { MarketplaceResponse } from "../claude_code_plugins/types";
import { ModelDataTable } from "../model_dashboard/table";
import NotificationsManager from "../molecules/notifications_manager";
import { getClaudeCodeMarketplace } from "../networking";
import { getMarketplaceTableColumns } from "./marketplace_table_columns";

interface ClaudeCodeMarketplaceTabProps {
  publicPage?: boolean;
}

const ClaudeCodeMarketplaceTab: React.FC<ClaudeCodeMarketplaceTabProps> = ({
  publicPage = false,
}) => {
  const [marketplaceData, setMarketplaceData] =
    useState<MarketplaceResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");

  useEffect(() => {
    const fetchMarketplace = async () => {
      setIsLoading(true);
      try {
        const data: MarketplaceResponse = await getClaudeCodeMarketplace();
        console.log("Claude Code marketplace:", data);
        setMarketplaceData(data);
      } catch (error) {
        console.error("Error fetching marketplace:", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchMarketplace();
  }, []);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  const categories = useMemo(() => {
    if (!marketplaceData) return ["All"];
    return extractCategories(marketplaceData.plugins);
  }, [marketplaceData]);

  const filteredPlugins = useMemo(() => {
    if (!marketplaceData) return [];

    let plugins = marketplaceData.plugins;
    plugins = filterPluginsByCategory(plugins, selectedCategory);
    plugins = filterPluginsBySearch(plugins, searchTerm);

    return plugins;
  }, [marketplaceData, selectedCategory, searchTerm]);

  const columns = useMemo(
    () => getMarketplaceTableColumns(copyToClipboard, publicPage),
    [publicPage],
  );

  if (!marketplaceData && !isLoading) {
    return (
      <Card>
        <div className="text-center p-12">
          <span className="text-muted-foreground">
            Failed to load marketplace. Please try again later.
          </span>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="max-w-md relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
        <Input
          placeholder="Search plugins by name, description, or keywords..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-9 h-11"
        />
      </div>

      <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
        <TabsList className="mb-4">
          {categories.map((category) => {
            const categoryPlugins = filterPluginsByCategory(
              marketplaceData?.plugins || [],
              category,
            );
            const count = filterPluginsBySearch(
              categoryPlugins,
              searchTerm,
            ).length;

            return (
              <TabsTrigger key={category} value={category}>
                {category} {count > 0 && `(${count})`}
              </TabsTrigger>
            );
          })}
        </TabsList>

        {categories.map((category) => (
          <TabsContent key={category} value={category}>
            <Card>
              <ModelDataTable
                columns={columns}
                data={filteredPlugins}
                isLoading={isLoading}
                defaultSorting={[{ id: "name", desc: false }]}
              />
            </Card>

            <div className="mt-4 text-center space-y-2">
              <span className="text-sm text-muted-foreground">
                Showing {filteredPlugins.length} of{" "}
                {marketplaceData?.plugins.length || 0} plugin
                {marketplaceData?.plugins.length !== 1 ? "s" : ""}
                {searchTerm && ` matching "${searchTerm}"`}
                {selectedCategory !== "All" && ` in ${selectedCategory}`}
              </span>
            </div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
};

export default ClaudeCodeMarketplaceTab;
