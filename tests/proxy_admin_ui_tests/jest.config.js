module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'node',
    transform: {
      '^.+\\.(ts|tsx)$': [
        'ts-jest',
        {
          tsconfig: 'tsconfig.json',
          jsx: 'react'
        }
      ]
    },
    moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node']
  };