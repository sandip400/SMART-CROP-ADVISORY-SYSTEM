import os
import pandas as pd

root = "Data_separate_files_header_20240830_20250910_12468_Qj2f_20250830"
out_file = "soil_recommendations.csv"

all_rows = []

# ---------- Recommendation Rules ----------
def recommend(row):
    moisture = row["soil_moisture"]
    soil = row["soil_type"]

    # Sandy soil rules
    if soil == "sand":
        if moisture < 15:
            return "Add compost and mulch to retain water"
        elif 15 <= moisture <= 40:
            return "Use drip irrigation to maintain moisture"
        else:
            return "Grow drought-tolerant crops and monitor salinity"

    # Clay soil rules
    elif soil == "clay":
        if moisture < 30:
            return "Mix in organic matter and compost to improve aeration"
        elif 30 <= moisture <= 60:
            return "Use raised beds to balance drainage and water retention"
        else:
            return "Improve drainage, add gypsum, and avoid over-irrigation"

    # Loamy soil rules
    elif soil == "loam":
        if moisture < 25:
            return "Add green manure and mulching to conserve moisture"
        elif 25 <= moisture <= 55:
            return "Maintain fertility with crop rotation and balanced irrigation"
        else:
            return "Ensure proper drainage, avoid waterlogging with cover crops"

    # Silty soil rules
    elif soil == "silty":
        if moisture < 20:
            return "Use compost to improve structure and water retention"
        elif 20 <= moisture <= 50:
            return "Use cover crops to prevent erosion and conserve water"
        else:
            return "Improve drainage with organic matter and controlled tillage"

    # Default fallback
    else:
        return "Apply organic matter and monitor irrigation schedule"


# ---------- Loop through dataset ----------
for network in os.listdir(root):
    network_path = os.path.join(root, network)
    if not os.path.isdir(network_path):
        continue

    for station in os.listdir(network_path):
        station_path = os.path.join(network_path, station)
        if not os.path.isdir(station_path):
            continue

        for file in os.listdir(station_path):
            if file.endswith(".stm"):
                filepath = os.path.join(station_path, file)
                print("✅ Processing:", filepath)

                try:
                    # Read stm file (auto-detect columns)
                    df = pd.read_csv(filepath, sep=r"\s+", comment='#', header=None)

                    # Detect number of columns
                    ncols = df.shape[1]
                    print(f"ℹ️ File {file} has {ncols} columns")

                    # Assign flexible column names
                    if ncols >= 3:
                        base_cols = ["date", "time", "soil_moisture"]
                        extra_cols = [f"extra_{i}" for i in range(ncols - len(base_cols))]
                        df.columns = base_cols + extra_cols
                    else:
                        print(f"⚠️ Skipping {file}, unexpected column count: {ncols}")
                        continue

                    # Add metadata
                    df["soil_type"] = "clay"    # TODO: parse from static_variables.csv
                    df["region"] = network

                    # Apply recommendation
                    df["improvement_step"] = df.apply(recommend, axis=1)

                    all_rows.append(df)

                except Exception as e:
                    print("⚠️ Skipped file:", filepath, "| Error:", e)


# ---------- Save final dataset ----------
if all_rows:
    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(out_file, index=False)
    print("✅ Saved:", out_file, "| Rows:", len(final_df))
else:
    print("⚠️ No valid .stm files were processed.")
