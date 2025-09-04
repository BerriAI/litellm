import React from 'react';

interface RaypathLogoProps {
  className?: string;
  textSize?: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
}

export const RaypathLogo: React.FC<RaypathLogoProps> = ({
  className = '',
  textSize = 'xl'
}) => {
  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
    xl: 'text-xl',
    '2xl': 'text-2xl'
  };

  return (
    <span className={`font-semibold ${textSizeClasses[textSize]} text-gray-900 ${className}`}>
      Raypath
    </span>
  );
};
