import React, { createContext, useContext, useState } from 'react';

interface FemaleFriendlyContextType {
  femaleFriendlyMode: boolean;
  setFemaleFriendlyMode: (value: boolean) => void;
}

const FemaleFriendlyContext = createContext<FemaleFriendlyContextType | undefined>(undefined);

export const FemaleFriendlyProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [femaleFriendlyMode, setFemaleFriendlyMode] = useState(false);

  return (
    <FemaleFriendlyContext.Provider value={{ femaleFriendlyMode, setFemaleFriendlyMode }}>
      {children}
    </FemaleFriendlyContext.Provider>
  );
};

export const useFemaleFriendly = () => {
  const context = useContext(FemaleFriendlyContext);
  if (context === undefined) {
    throw new Error('useFemaleFriendly must be used within a FemaleFriendlyProvider');
  }
  return context;
}; 