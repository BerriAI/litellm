import sys
import filecmp


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print("Usage: python check_files_match.py <file1> <file2>")
        return 1

    file1 = argv[0]
    file2 = argv[1]

    if filecmp.cmp(file1, file2, shallow=False):
        print(f"Files {file1} and {file2} match.")
        return 0
    else:
        print(f"Files {file1} and {file2} do not match.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
