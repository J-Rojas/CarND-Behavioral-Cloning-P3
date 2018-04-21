import sys
import model

if len(sys.argv) < 5:
    sys.exit("Error in arguments: weights_file train_csv test_csv train_dir [validate_dir] [save_file] [epochs]")

weights_file = sys.argv[1]
train_csv = sys.argv[2]
test_csv = sys.argv[3]
train_dir = sys.argv[4]
validate_dir = None
if len(sys.argv) > 5:
    validate_dir = sys.argv[5]

save_file = None
if len(sys.argv) > 6:
    save_file = sys.argv[6]

epochs = None
if len(sys.argv) > 7:
    epochs = int(sys.argv[7])

model.loadAndTrain(train_csv, test_csv, weights_file, train_dir, validate_dir, \
    {'flip_images': True, 'dropout_rate': 0.5, 'use_left_right': True, 'use_trans': 20, 'save_file': save_file, 'epochs': epochs})
