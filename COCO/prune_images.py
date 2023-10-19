import os

def prune_unmatched_files(dir_to_prune, reference_dir):
  
    files_in_dir_to_prune = set(os.path.splitext(filename)[0] for filename in os.listdir(dir_to_prune))
    # print(files_in_dir_to_prune)
    files_in_reference_dir = set(os.path.splitext(filename)[0][:-6] for filename in os.listdir(reference_dir))
    # print(files_in_reference_dir)

    unmatched_files = files_in_dir_to_prune.difference(files_in_reference_dir)

    print(len(unmatched_files))

    for filename in unmatched_files:
        # print(filename)
        file_path = os.path.join(dir_to_prune, filename)
        file_path = file_path + ".jpg"
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    curr_dir = os.getcwd()
    dir_to_prune_path = os.path.join(curr_dir, 'val2017')
    reference_dir_path = os.path.join(curr_dir, 'val2017_masks')

    prune_unmatched_files(dir_to_prune_path, reference_dir_path)