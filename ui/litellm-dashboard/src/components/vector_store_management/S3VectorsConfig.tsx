import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Alert, Form, Input, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";

interface S3VectorsConfigProps {
  accessToken: string | null;
  providerParams: Record<string, any>;
  onParamsChange: (params: Record<string, any>) => void;
}

const S3VectorsConfig: React.FC<S3VectorsConfigProps> = ({ accessToken, providerParams, onParamsChange }) => {
  const { t } = useTranslation();
  const [embeddingModels, setEmbeddingModels] = useState<ModelGroup[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  useEffect(() => {
    if (!accessToken) return;

    const loadModels = async () => {
      setIsLoadingModels(true);
      try {
        const models = await fetchAvailableModels(accessToken);
        // Filter for embedding models only
        const embeddingOnly = models.filter((model) => model.mode === "embedding");
        setEmbeddingModels(embeddingOnly);
      } catch (error) {
        console.error("Error fetching embedding models:", error);
      } finally {
        setIsLoadingModels(false);
      }
    };

    loadModels();
  }, [accessToken]);

  const handleFieldChange = (fieldName: string, value: string) => {
    onParamsChange({
      ...providerParams,
      [fieldName]: value,
    });
  };

  return (
    <>
      {/* S3 Vectors Setup Instructions */}
      <Alert
        message={t("vectorStoreManagement.s3VectorsConfig.alertTitle")}
        description={
          <div>
            <p>{t("vectorStoreManagement.s3VectorsConfig.alertDescription")}</p>
            <ul style={{ marginLeft: "16px", marginTop: "8px" }}>
              <li>{t("vectorStoreManagement.s3VectorsConfig.alertItem1")}</li>
              <li>{t("vectorStoreManagement.s3VectorsConfig.alertItem2")}</li>
              <li>{t("vectorStoreManagement.s3VectorsConfig.alertItem3")}</li>
              <li>
                {t("vectorStoreManagement.s3VectorsConfig.alertItem4Prefix")}{" "}
                <a
                  href="https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vector-buckets.html"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {t("vectorStoreManagement.s3VectorsConfig.alertItem4Link")}
                </a>
              </li>
            </ul>
          </div>
        }
        type="info"
        showIcon
        style={{ marginBottom: "16px" }}
      />

      {/* Vector Bucket Name */}
      <Form.Item
        label={
          <span>
            {t("vectorStoreManagement.s3VectorsConfig.vectorBucketNameLabel")}{" "}
            <Tooltip title={t("vectorStoreManagement.s3VectorsConfig.vectorBucketNameTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        required
        validateStatus={
          providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3 ? "error" : undefined
        }
        help={
          providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3
            ? t("vectorStoreManagement.s3VectorsConfig.bucketNameHelp")
            : undefined
        }
      >
        <Input
          value={providerParams.vector_bucket_name || ""}
          onChange={(e) => handleFieldChange("vector_bucket_name", e.target.value)}
          placeholder={t("vectorStoreManagement.s3VectorsConfig.vectorBucketNamePlaceholder")}
          size="large"
          className="rounded-md"
        />
      </Form.Item>

      {/* Index Name (Optional) */}
      <Form.Item
        label={
          <span>
            {t("vectorStoreManagement.s3VectorsConfig.indexNameLabel")}{" "}
            <Tooltip title={t("vectorStoreManagement.s3VectorsConfig.indexNameTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        validateStatus={
          providerParams.index_name && providerParams.index_name.length > 0 && providerParams.index_name.length < 3
            ? "error"
            : undefined
        }
        help={
          providerParams.index_name && providerParams.index_name.length > 0 && providerParams.index_name.length < 3
            ? t("vectorStoreManagement.s3VectorsConfig.indexNameHelp")
            : undefined
        }
      >
        <Input
          value={providerParams.index_name || ""}
          onChange={(e) => handleFieldChange("index_name", e.target.value)}
          placeholder={t("vectorStoreManagement.s3VectorsConfig.indexNamePlaceholder")}
          size="large"
          className="rounded-md"
        />
      </Form.Item>

      {/* AWS Region */}
      <Form.Item
        label={
          <span>
            {t("vectorStoreManagement.s3VectorsConfig.awsRegionLabel")}{" "}
            <Tooltip title={t("vectorStoreManagement.s3VectorsConfig.awsRegionTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        required
      >
        <Input
          value={providerParams.aws_region_name || ""}
          onChange={(e) => handleFieldChange("aws_region_name", e.target.value)}
          placeholder="us-west-2"
          size="large"
          className="rounded-md"
        />
      </Form.Item>

      {/* Embedding Model */}
      <Form.Item
        label={
          <span>
            {t("vectorStoreManagement.s3VectorsConfig.embeddingModelLabel")}{" "}
            <Tooltip title={t("vectorStoreManagement.s3VectorsConfig.embeddingModelTooltip")}>
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        required
      >
        <Select
          value={providerParams.embedding_model || undefined}
          onChange={(value) => handleFieldChange("embedding_model", value)}
          placeholder={t("vectorStoreManagement.s3VectorsConfig.embeddingModelPlaceholder")}
          size="large"
          showSearch
          loading={isLoadingModels}
          filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
          options={embeddingModels.map((model) => ({
            value: model.model_group,
            label: model.model_group,
          }))}
          style={{ width: "100%" }}
        />
      </Form.Item>
    </>
  );
};

export default S3VectorsConfig;
