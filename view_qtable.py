import numpy as np

# Define the filename of your saved Q-table
filename = 'qtable_layout_rl.npy'

try:
    # Load the Q-table from the .npy file
    q_table = np.load(filename)

    # Print basic information about the Q-table
    print(f"Successfully loaded Q-table from: {filename}")
    print(f"Q-table shape (states x actions): {q_table.shape}")
    print(f"Data type: {q_table.dtype}")

    # Print the full contents of the Q-table
    print("\nQ-table contents:")
    print(q_table)

    # You can also use other numpy functions to analyze the data, e.g.,
    # print("\nMax Q-value:", np.max(q_table))
    # print("Min Q-value:", np.min(q_table))

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    print("Please make sure the file path is correct.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")