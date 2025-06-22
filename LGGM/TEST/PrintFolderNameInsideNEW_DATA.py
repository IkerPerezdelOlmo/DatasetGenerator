import os

# This program prints a list of all folder names inside NEW_DATA so that you can paset them where neccesary
def get_subfolder_names(directory):
    """
    Gets the names of all subfolders within a given directory.

    Args:
        directory (str): The path to the directory to scan.

    Returns:
        list: A list of strings, where each string is the name of a subfolder.
              Returns an empty list if the directory doesn't exist or has no subfolders.
    """
    subfolder_names = []
    if os.path.isdir(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                subfolder_names.append(item)
    return subfolder_names

if __name__ == "__main__":
    # You can change this to the directory you want to scan
    target_directory = "./NEW_DATA"  # Current directory

    folders = get_subfolder_names(target_directory)
    print(folders)
    print(len(folders))

    # Example with a different directory (uncomment and modify to test)
    # target_directory_example = "./OLD_DATA"
    # folders_example = get_subfolder_names(target_directory_example)
    # print(folders_example)