# /index-docs Task

When this command is used, execute the following task:

<!-- Powered by BMAD™ Core -->

# Index Documentation Task

## Purpose

This task maintains the integrity and completeness of the `docs/index.md` file by scanning all documentation files and ensuring they are properly indexed with descriptions. It handles both root-level documents and documents within subfolders, organizing them hierarchically.

## Task Instructions

You are now operating as a Documentation Indexer. Your goal is to ensure all documentation files are properly cataloged in the central index with proper organization for subfolders.

### Required Steps

1. First, locate and scan:
   - The `docs/` directory and all subdirectories
   - The existing `docs/index.md` file (create if absent)
   - All markdown (`.md`) and text (`.txt`) files in the documentation structure
   - Note the folder structure for hierarchical organization

2. For the existing `docs/index.md`:
   - Parse current entries
   - Note existing file references and descriptions
   - Identify any broken links or missing files
   - Keep track of already-indexed content
   - Preserve existing folder sections

3. For each documentation file found:
   - Extract the title (from first heading or filename)
   - Generate a brief description by analyzing the content
   - Create a relative markdown link to the file
   - Check if it's already in the index
   - Note which folder it belongs to (if in a subfolder)
   - If missing or outdated, prepare an update

4. For any missing or non-existent files found in index:
   - Present a list of all entries that reference non-existent files
   - For each entry:
     - Show the full entry details (title, path, description)
     - Ask for explicit confirmation before removal
     - Provide option to update the path if file was moved
     - Log the decision (remove/update/keep) for final report

5. Update `docs/index.md`:
   - Maintain existing structure and organization
   - Create level 2 sections (`##`) for each subfolder
   - List root-level documents first
   - Add missing entries with descriptions
   - Update outdated entries
   - Remove only entries that were confirmed for removal
   - Ensure consistent formatting throughout

### Index Structure Format

The index should be organized as follows:

```markdown
# Documentation Index

## Root Documents

### [Document Title](./document.md)

Brief description of the document's purpose and contents.

### [Another Document](./another.md)

Description here.

## Folder Name

Documents within the `folder-name/` directory:

### [Document in Folder](./folder-name/document.md)

Description of this document.

### [Another in Folder](./folder-name/another.md)

Description here.

## Another Folder

Documents within the `another-folder/` directory:

### [Nested Document](./another-folder/document.md)

Description of nested document.
```

### Index Entry Format

Each entry should follow this format:

```markdown
### [Document Title](relative/path/to/file.md)

Brief description of the document's purpose and contents.
```

### Rules of Operation

1. NEVER modify the content of indexed files
2. Preserve existing descriptions in index.md when they are adequate
3. Maintain any existing categorization or grouping in the index
4. Use relative paths for all links (starting with `./`)
5. Ensure descriptions are concise but informative
6. NEVER remove entries without explicit confirmation
7. Report any broken links or inconsistencies found
8. Allow path updates for moved files before considering removal
9. Create folder sections using level 2 headings (`##`)
10. Sort folders alphabetically, with root documents listed first
11. Within each section, sort documents alphabetically by title

### Process Output

The task will provide:

1. A summary of changes made to index.md
2. List of newly indexed files (organized by folder)
3. List of updated entries
4. List of entries presented for removal and their status:
   - Confirmed removals
   - Updated paths
   - Kept despite missing file
5. Any new folders discovered
6. Any other issues or inconsistencies found

### Handling Missing Files

For each file referenced in the index but not found in the filesystem:

1. Present the entry:

   ```markdown
   Missing file detected:
   Title: [Document Title]
   Path: relative/path/to/file.md
   Description: Existing description
   Section: [Root Documents | Folder Name]

   Options:

   1. Remove this entry
   2. Update the file path
   3. Keep entry (mark as temporarily unavailable)

   Please choose an option (1/2/3):
   ```

2. Wait for user confirmation before taking any action
3. Log the decision for the final report

### Special Cases

1. **Sharded Documents**: If a folder contains an `index.md` file, treat it as a sharded document:
   - Use the folder's `index.md` title as the section title
   - List the folder's documents as subsections
   - Note in the description that this is a multi-part document

2. **README files**: Convert `README.md` to more descriptive titles based on content

3. **Nested Subfolders**: For deeply nested folders, maintain the hierarchy but limit to 2 levels in the main index. Deeper structures should have their own index files.

## Required Input

Please provide:

1. Location of the `docs/` directory (default: `./docs`)
2. Confirmation of write access to `docs/index.md`
3. Any specific categorization preferences
4. Any files or directories to exclude from indexing (e.g., `.git`, `node_modules`)
5. Whether to include hidden files/folders (starting with `.`)

Would you like to proceed with documentation indexing? Please provide the required input above.
