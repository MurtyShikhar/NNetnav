# Write a script that go through all the folders under
# gpt-4o, and load the summary.json file in each folder
# and print the sum of the values of cum_reward

if __name__ == "__main__":
    import os
    import json

    # Path to the directory containing the folders
    base_dir = "./gpt-4o/"

    # Initialize a variable to hold the

    total_cum_reward = 0

    # Iterate through each folder in the base directory
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            summary_file_path = os.path.join(folder_path, "summary_info.json")

            # Check if the summary.json file exists
            if os.path.isfile(summary_file_path):
                with open(summary_file_path, "r") as f:
                    data = json.load(f)
                    total_cum_reward += data.get("cum_reward", 0)
                    # Print the cum_reward for this folder
                    print(f"Folder: {folder}, cum_reward: {data.get('cum_reward', 0)}")
        
    # Print the total cum_reward
    print(f"Total cum_reward: {total_cum_reward}")