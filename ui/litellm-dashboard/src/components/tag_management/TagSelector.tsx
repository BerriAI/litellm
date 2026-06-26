import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { useTranslation } from "react-i18next";
import { Tag } from "./types";
import { tagListCall } from "../networking";

interface TagSelectorProps {
  onChange: (selectedTags: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
}

const TagSelector: React.FC<TagSelectorProps> = ({ onChange, value, className, accessToken }) => {
  const { t } = useTranslation();
  const [tags, setTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchTags = async () => {
      if (!accessToken) return;
      try {
        const response = await tagListCall(accessToken);
        console.log("List tags response:", response);
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
    <Select
      mode="tags"
      showSearch
      placeholder={t("tagManagement.tagSelector.placeholder")}
      onChange={onChange}
      value={value}
      loading={loading}
      className={className}
      options={tags.map((tag) => ({
        label: tag.name,
        value: tag.name,
        title: tag.description || tag.name,
      }))}
      optionFilterProp="label"
      tokenSeparators={[","]}
      maxTagCount="responsive"
      allowClear
      style={{ width: "100%" }}
    />
  );
};

export default TagSelector;
