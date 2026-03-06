import React, { useEffect, useState } from "react";
import { Select, Skeleton } from "antd";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";

const LOCALSTORAGE_KEY = "litellm_chat_selected_model";
const MAX_DISPLAY_LENGTH = 40;

interface Props {
  accessToken: string;
  selectedModel: string;
  onChange: (model: string) => void;
  onLoadingChange: (loading: boolean) => void;
}

const ModelSelector: React.FC<Props> = ({
  accessToken,
  selectedModel,
  onChange,
  onLoadingChange,
}) => {
  const [models, setModels] = useState<ModelGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchFailed, setFetchFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      onLoadingChange(true);
      try {
        const fetched = await fetchAvailableModels(accessToken);
        if (cancelled) return;

        setModels(fetched);

        if (fetched.length > 0) {
          const persisted = localStorage.getItem(LOCALSTORAGE_KEY);
          const modelNames = fetched.map((m) => m.model_group);

          if (persisted && modelNames.includes(persisted)) {
            onChange(persisted);
          } else {
            // Persisted model not in list — clear stale value and default to first
            if (persisted) {
              localStorage.removeItem(LOCALSTORAGE_KEY);
            }
            onChange(fetched[0].model_group);
          }
        }
      } catch {
        if (!cancelled) {
          setFetchFailed(true);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
          onLoadingChange(false);
        }
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, [accessToken]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleChange = (value: string) => {
    localStorage.setItem(LOCALSTORAGE_KEY, value);
    onChange(value);
  };

  if (loading) {
    return <Skeleton.Input active style={{ width: 220 }} />;
  }

  if (fetchFailed || models.length === 0) {
    return (
      <span style={{ color: "#8c8c8c", fontSize: 13 }}>
        No models available — check your proxy config
      </span>
    );
  }

  return (
    <Select
      value={selectedModel}
      onChange={handleChange}
      style={{ width: 220 }}
      showSearch
      filterOption={(input, option) =>
        (option?.label as string ?? "").toLowerCase().includes(input.toLowerCase())
      }
      options={models.map((m) => ({
        value: m.model_group,
        label:
          m.model_group.length > MAX_DISPLAY_LENGTH
            ? `${m.model_group.slice(0, MAX_DISPLAY_LENGTH)}…`
            : m.model_group,
      }))}
    />
  );
};

export default ModelSelector;
