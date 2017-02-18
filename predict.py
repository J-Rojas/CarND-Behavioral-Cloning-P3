import sys
import model

if len(sys.argv) < 4:
    sys.exit("Error in arguments: csv_file weights_file validate_dir")

csv_file = sys.argv[1]
weights_file = sys.argv[2]
validate_dir = sys.argv[3]

model.predict(csv_file, weights_file, validate_dir)
