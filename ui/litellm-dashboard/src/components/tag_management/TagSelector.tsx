import React, { useEffect, useState } from "react";
import { Tag } from "./types";
import { tagListCall } from "../networking";
import { MultiSelect } from "@/components/shared/MultiSelect";

interface TagSelectorProps {
  onChange: (selectedTags: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
}

const TagSelector: React.FC<TagSelectorProps> = ({ onChange, value, className, accessToken }) => {
  const [tags, setTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchTags = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const response = await tagListCall(accessToken);
        setTags(Object.values(response));
      } catch (error) {
        console.error("Error fetching tags:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchTags();
  }, [accessToken]);

  return (
    <MultiSelect
      placeholder="Select or create tags"
      onValueChange={onChange}
      value={value}
      loading={loading}
      className={className}
      allowCustomValues
      options={tags.map((tag) => ({
        label: tag.name,
        value: tag.name,
        description: tag.description || undefined,
      }))}
    />
  );
};

export default TagSelector;
