import { useState } from "react";

const useFeatureFlags = () => {
  const [refactoredUIFlag, setRefactoredUIFlag] = useState<boolean>(false);

  return { refactoredUIFlag, setRefactoredUIFlag };
};

export default useFeatureFlags;
