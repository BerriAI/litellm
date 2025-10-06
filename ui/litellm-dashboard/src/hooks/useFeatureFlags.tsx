'use client';
import React, {createContext, useContext, useState} from 'react';

type Flags = {
  refactoredUIFlag: boolean;
  setRefactoredUIFlag: (v: boolean) => void;
};

const FeatureFlagsCtx = createContext<Flags | null>(null);

export const FeatureFlagsProvider = ({ children }: { children: React.ReactNode }) => {
  const [refactoredUIFlag, setRefactoredUIFlag] = useState(false);
  return (
    <FeatureFlagsCtx.Provider value={{ refactoredUIFlag, setRefactoredUIFlag }}>
    {children}
    </FeatureFlagsCtx.Provider>
);
}

const useFeatureFlags = () => {
  const ctx = useContext(FeatureFlagsCtx);
  if (!ctx) throw new Error('useFeatureFlags must be used within FeatureFlagsProvider');
  return ctx;
}

export default useFeatureFlags;
