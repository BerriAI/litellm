interface ModelNameDisplayProps {
  model: {
    model_name?: string;
    model_info?: {
      team_public_model_name?: string;
    };
  };
}

export const getDisplayModelName = (model: ModelNameDisplayProps["model"]): string => {
  if (model?.model_info?.team_public_model_name) {
    return model.model_info.team_public_model_name;
  }
  return model?.model_name || "-";
};
