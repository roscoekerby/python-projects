import os


def search_file(directory, target_file):
    found_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == target_file:
                found_paths.append(os.path.join(root, file))
    return found_paths


def main():
    directory = input("Enter the directory to search: ")
    target_file = input("Enter the file name to search for: ")


    found_paths = search_file(directory, target_file)


    if found_paths:
        print(f"Found {len(found_paths)} occurrences of '{target_file}':")
        for path in found_paths:
            print(path)
    else:
        print(f"No occurrences of '{target_file}' found in the specified directory.")


if __name__ == "__main__":
    main()
