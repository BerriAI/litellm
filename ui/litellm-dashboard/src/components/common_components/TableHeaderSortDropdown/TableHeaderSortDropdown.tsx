import React from "react";
import { Button, Dropdown, MenuProps } from "antd";
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon, XIcon } from "@heroicons/react/outline";

export type SortState = "asc" | "desc" | false;

interface TableHeaderSortDropdownProps {
  /**
   * Current sort state: "asc", "desc", or false for neutral
   */
  sortState: SortState;
  /**
   * Callback when sort state changes
   * @param newState - The new sort state: "asc", "desc", or false
   */
  onSortChange: (newState: SortState) => void;
  /**
   * Optional column ID for identification
   */
  columnId?: string;
}

export const TableHeaderSortDropdown: React.FC<TableHeaderSortDropdownProps> = ({
  sortState,
  onSortChange,
}) => {
  const handleMenuClick: MenuProps["onClick"] = ({ key }) => {
    if (key === "asc") {
      onSortChange("asc");
    } else if (key === "desc") {
      onSortChange("desc");
    } else if (key === "reset") {
      onSortChange(false);
    }
  };

  const menuItems: MenuProps["items"] = [
    {
      key: "asc",
      label: "Ascending",
      icon: <ChevronUpIcon className="h-4 w-4" />,
    },
    {
      key: "desc",
      label: "Descending",
      icon: <ChevronDownIcon className="h-4 w-4" />,
    },
    {
      key: "reset",
      label: "Reset",
      icon: <XIcon className="h-4 w-4" />,
    },
  ];

  // Determine which icon to display based on current sort state
  const renderIcon = () => {
    if (sortState === "asc") {
      return <ChevronUpIcon className="h-4 w-4" />;
    } else if (sortState === "desc") {
      return <ChevronDownIcon className="h-4 w-4" />;
    } else {
      return <SwitchVerticalIcon className="h-4 w-4" />;
    }
  };

  return (
    <Dropdown
      menu={{
        items: menuItems,
        onClick: handleMenuClick,
        selectable: true,
        selectedKeys: sortState ? [sortState] : [],
      }}
      trigger={["click"]}
      autoAdjustOverflow
    >
      <Button
        type="text"
        onClick={(e) => e.stopPropagation()}
        icon={renderIcon()}
        className={sortState ? "text-blue-500 hover:text-blue-600" : "text-gray-400 hover:text-blue-500"}
      />
    </Dropdown>
  );
};
