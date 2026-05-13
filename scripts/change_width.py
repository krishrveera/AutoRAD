import xml.etree.ElementTree as ET
import argparse
import os

def widen_lanes(input_file, new_width_meters):
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        return

    print(f"Scanning '{input_file}' to widen driving lanes to {new_width_meters}m...")
    
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        widths_changed = 0
        
        # Search for every lane in the file
        for lane in root.findall('.//lane'):
            # We only want to widen the drivable lanes, not sidewalks or shoulders
            if lane.get('type') == 'driving':
                # Find the width tags inside this lane
                for width in lane.findall('width'):
                    # Update the 'a' attribute to the new width
                    width.set('a', str(float(new_width_meters)))
                    widths_changed += 1
                    
        # Create a dynamic output file name based on the input and width
        file_name, file_extension = os.path.splitext(input_file)
        output_file = f"{file_name}_{new_width_meters}m{file_extension}"
        
        # Save the new file
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"Success! Widened {widths_changed} lane sections.")
        print(f"Saved new map as: {output_file}")
        
    except Exception as e:
        print(f"Failed to process file: {e}")

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Widen the driving lanes of a CARLA OpenDRIVE (.xodr) map.")
    
    # Add the required arguments
    parser.add_argument(
        "file", 
        type=str, 
        help="The path to your .xodr file (e.g., racetrack_fixed.xodr)"
    )
    parser.add_argument(
        "width", 
        type=float, 
        help="The new width for the driving lanes in meters (e.g., 6.0)"
    )
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Execute the function with the parsed arguments
    widen_lanes(args.file, args.width)