module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'jsdom',
    moduleNameMapper: {
      '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
      '\\.(jpg|jpeg|png|gif|webp|svg)$': '<rootDir>/__mocks__/fileMock.js'
    },
    setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
    testMatch: [
      '<rootDir>/**/*.test.tsx',
      '<rootDir>/**/*_test.tsx'  // Added this to match your file naming
    ],
    moduleDirectories: ['node_modules'],
    testPathIgnorePatterns: ['/node_modules/'],
    transform: {
      '^.+\\.(ts|tsx)$': 'ts-jest'
    }
  }