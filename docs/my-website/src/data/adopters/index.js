import adoptersData from './adopters.json';

/**
 * @typedef {Object} Adopter
 * @property {string} name - The organization's display name
 * @property {string} logoUrl - URL to the organization's logo
 * @property {string} [url] - The organization's website URL
 * @property {string} [description] - Brief description shown on hover
 */

/**
 * List of organizations using LiteLLM
 * @type {Adopter[]}
 */
export const adopters = adoptersData;

/**
 * Adopters sorted alphabetically by name
 * @type {Adopter[]}
 */
export const sortedAdopters = [...adopters].sort((a, b) => 
  a.name.localeCompare(b.name)
);
