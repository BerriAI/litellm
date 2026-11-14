import sys


def check_file_length(max_lines, filenames):
    bad_files = []
    for filename in filenames:
        with open(filename, "r") as file:
            lines = file.readlines()
            if len(lines) > max_lines:
                bad_files.append((filename, len(lines)))
    return bad_files


if __name__ == "__main__":
    max_lines = int(sys.argv[1])
    filenames = sys.argv[2:]

    bad_files = check_file_length(max_lines, filenames)
    if bad_files:
        bad_files.sort(
            key=lambda x: x[1], reverse=True
        )  # Sort files by length in descending order
        for filename, length in bad_files:
            print(f"{filename}: {length} lines")

        sys.exit(1)
    else:
        sys.exit(0)
