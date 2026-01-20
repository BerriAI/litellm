import { SearchOutlined } from "@ant-design/icons";
import { Card, Tab, TabGroup, TabList, TabPanel, TabPanels, Text } from "@tremor/react";
import { Input } from "antd";
import React, { useEffect, useMemo, useState } from "react";
import {
  extractCategories,
  filterPluginsByCategory,
  filterPluginsBySearch,
} from "../claude_code_plugins/helpers";
import {
  MarketplaceResponse
} from "../claude_code_plugins/types";
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
  const [selectedCategoryIndex, setSelectedCategoryIndex] = useState(0);

  useEffect(() => {
    fetchMarketplace();
  }, []);

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

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  // Extract unique categories from plugins
  const categories = useMemo(() => {
    if (!marketplaceData) return ["All"];
    return extractCategories(marketplaceData.plugins);
  }, [marketplaceData]);

  // Get selected category name
  const selectedCategory = categories[selectedCategoryIndex] || "All";

  // Filter plugins by search and category
  const filteredPlugins = useMemo(() => {
    if (!marketplaceData) return [];

    let plugins = marketplaceData.plugins;

    // Apply category filter
    plugins = filterPluginsByCategory(plugins, selectedCategory);

    // Apply search filter
    plugins = filterPluginsBySearch(plugins, searchTerm);

    return plugins;
  }, [marketplaceData, selectedCategory, searchTerm]);

  const columns = useMemo(
    () => getMarketplaceTableColumns(copyToClipboard, publicPage),
    [publicPage]
  );

  if (!marketplaceData && !isLoading) {
    return (
      <Card>
        <div className="text-center p-12">
          <Text className="text-gray-500">
            Failed to load marketplace. Please try again later.
          </Text>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="max-w-md">
        <Input
          placeholder="Search plugins by name, description, or keywords..."
          prefix={<SearchOutlined className="text-gray-400" />}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          allowClear
          size="large"
        />
      </div>

      {/* Category Tabs */}
      <TabGroup index={selectedCategoryIndex} onIndexChange={setSelectedCategoryIndex}>
        <TabList className="mb-4">
          {categories.map((category) => {
            // Count plugins in this category
            const categoryPlugins = filterPluginsByCategory(
              marketplaceData?.plugins || [],
              category
            );
            const count = filterPluginsBySearch(
              categoryPlugins,
              searchTerm
            ).length;

            return (
              <Tab key={category}>
                {category} {count > 0 && `(${count})`}
              </Tab>
            );
          })}
        </TabList>

        <TabPanels>
          {categories.map((category) => (
            <TabPanel key={category}>
              <Card>
                {/* Plugin Table */}
                <ModelDataTable
                  columns={columns}
                  data={filteredPlugins}
                  isLoading={isLoading}
                  defaultSorting={[{ id: "name", desc: false }]}
                />
              </Card>

              {/* Footer Info */}
              <div className="mt-4 text-center space-y-2">
                <Text className="text-sm text-gray-600">
                  Showing {filteredPlugins.length} of{" "}
                  {marketplaceData?.plugins.length || 0} plugin
                  {marketplaceData?.plugins.length !== 1 ? "s" : ""}
                  {searchTerm && ` matching "${searchTerm}"`}
                  {selectedCategory !== "All" && ` in ${selectedCategory}`}
                </Text>
              </div>
            </TabPanel>
          ))}
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default ClaudeCodeMarketplaceTab;
