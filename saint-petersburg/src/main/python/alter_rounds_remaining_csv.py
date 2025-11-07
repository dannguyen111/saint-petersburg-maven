import pandas as pd
import numpy as np
import io

# Load the CSV file
file_path = "rounds_remaining.csv"
try:
    df = pd.read_csv(file_path)

    print(f"Successfully loaded {file_path}.")
    print("\n--- Original Columns and Head (First 5 Rows) ---")
    print(df.info())
    print(df.head())

    # Ensure the required columns exist
    if 'current_round' not in df.columns or 'current_phase' not in df.columns:
        print(f"Error: The CSV must contain 'current_round' and 'current_phase' columns.")
    else:
        # Get the two columns
        phase = df['current_phase']
        round_num = df['current_round']

        # 1. Implement the Java logic for 'currPhase' using np.where
        # int currPhase = (state.phase < 4) ? state.phase : (state.phase == 4) ? 2 : 0;
        curr_phase = np.where(phase < 4, phase, np.where(phase == 4, 2, 0))

        # 2. Implement the Java logic for 'round'
        # double round = (double) state.round + (currPhase % 4) / 4.0;
        # We will overwrite the 'current_round' column
        df['current_round'] = round_num + (curr_phase % 4) / 4.0

        # 3. Drop the 'current_phase' column as it's now combined
        df = df.drop(columns=['current_phase'])

        # 4. Save the new DataFrame to a new CSV file
        output_filename = "rounds_remaining_combined.csv"
        df.to_csv(output_filename, index=False)

        print(f"\n--- Transformation Complete ---")
        print(f"✅ 'current_round' and 'current_phase' have been combined.")
        print(f"✅ 'current_phase' column has been dropped.")
        print(f"✅ New file saved as: {output_filename}")

        print("\n--- New DataFrame Head (First 5 Rows) ---")
        print(df.head())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")