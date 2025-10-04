import os
import sys # Added for cleaner exit

# --- Configuration for THIRD RUN (VALIDATION LABELS) ---
# 1. SET THE PATH TO YOUR ANNOTATION FOLDER
LABEL_DIR = 'strap_id_dataset/valid/labels'

# 2. IMAGE SIZE (Usually not needed if input is already normalized, but good to keep)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640


def convert_polygon_to_bbox(line):
    """
    Converts a single line of polygon data to YOLO bounding box format.
    
    Input line format (normalized): class x1 y1 x2 y2 x3 y3 ...
    Output format (normalized): class x_center y_center width height
    """
    parts = line.strip().split()
    
    if not parts:
        return None

    try:
        class_id = int(parts[0])
        # Ensure coordinates are float and normalized (0-1)
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        # print(f"Skipping line due to non-numeric data: {line.strip()}")
        return None

    # Check if we have at least one valid x, y pair (2 pairs for a box, 4 coordinates total)
    if len(coords) < 4 or len(coords) % 2 != 0:
        # print(f"Skipping line: Polygon has incomplete or fewer than 2 vertices: {line.strip()}")
        return None

    # Separate x and y coordinates
    x_coords = coords[::2]
    y_coords = coords[1::2]

    # Calculate min/max for the Bounding Box
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # --- Convert to YOLO BBox Format ---
    
    # Bounding Box dimensions
    width = max_x - min_x
    height = max_y - min_y

    # Center coordinates
    x_center = min_x + (width / 2)
    y_center = min_y + (height / 2)

    # Ensure all values are within 0-1 range (safety check)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    # Format output as YOLO Bounding Box string (6 decimal places is standard)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_labels_directory(label_dir):
    """Iterates through all .txt files and converts them."""
    print(f"--- Starting Annotation Conversion ---")
    print(f"Target Directory: {os.path.abspath(label_dir)}")
    
    if not os.path.exists(label_dir):
        print(f"\nERROR: The directory '{label_dir}' does not exist.")
        print("Please check and update the 'LABEL_DIR' variable in the script.")
        return

    processed_count = 0
    failed_count = 0
    
    # Loop through all files in the directory
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            input_filepath = os.path.join(label_dir, filename)
            
            try:
                with open(input_filepath, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                # Process each annotation line in the file
                for line in lines:
                    bbox_line = convert_polygon_to_bbox(line)
                    if bbox_line:
                        new_lines.append(bbox_line)
                
                # Overwrite the original file with the new bounding box data
                with open(input_filepath, 'w') as f:
                    # Write only if there are valid annotations, otherwise write an empty file
                    if new_lines:
                         f.write('\n'.join(new_lines) + '\n')
                
                processed_count += 1
            
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                failed_count += 1

    print("\n--- Conversion Complete ---")
    print(f"Total files processed successfully: {processed_count}")
    print(f"Total files failed: {failed_count}")
    print(f"\nAnnotations in {label_dir} are now in Bounding Box format.")


if __name__ == '__main__':
    # --- SAFETY WARNING AND CONFIRMATION ---
    print("!!! WARNING: This script will OVERWRITE your existing .txt label files in the selected folder. !!!")
    print("!!! Please ensure you have backed up your original annotations before proceeding. !!!")
    
    confirmation = input(f"Do you want to convert annotations in '{LABEL_DIR}'? Type 'YES' to proceed: ")
    
    if confirmation.upper() == 'YES':
        process_labels_directory(LABEL_DIR)
        print("\nNEXT STEP: Update the LABEL_DIR in this script to 'strap_id_dataset/test/labels' and run it again.")
    else:
        print("Conversion aborted. Please run the script again and type 'YES' to confirm.")
        sys.exit(0)