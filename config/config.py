import argparse

BETA = 0.005
BATCH_SIZE = 4
ITER_SIZE = 100
NUM_WORKERS = 4
INPUT_SIZE = '2048,1024'
INPUT_SIZE_RF = '1280,960'
NUM_EPOCHS = 30
NUM_STEPS = 1000000
NUM_EPOCHS_FPF = 1
NUM_STEPS_STOP = 60000  
RANDOM_SEED = 1234

SF_ROOT = "./dataset02/foggy"
CW_ROOT = "./dataset02/clear"
RF_ROOT = "./dataset02/Foggy_Driving/Processed_foggy_driving"
SET = 'train'

WEIGHT_BOX = 1
WEIGHT_CLS = 1
WEIGHT_OBJ = 1
WEIGHT_DFL = 1.5
WEIGHT_FSM = 1
WEIGHT_CON = 0.1

def get_arguments():

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--sf-root", type=str, default=SF_ROOT, help="Path to Cityscapes foggy dataset")
    parser.add_argument("--cw-root", type=str, default=CW_ROOT, help="Path to Cityscapes dataset")
    parser.add_argument("--rf-root", type=str, default=RF_ROOT, help="Path to Foggy Driving dataset")
    parser.add_argument("--set", type=str, default=SET, help="Dataset split (train/val/test)")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE, help="Iteration size")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of workers")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE, help="Input image size 'width,height'")
    parser.add_argument("--input-size-rf", type=str, default=INPUT_SIZE_RF, help="Input RF image size 'width,height'")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help="Number of training steps")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP, help="Number of steps for early stopping")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--beta", type=float, default=BETA, help="Beta parameter")
    parser.add_argument("--num-epochs-fpf", type=float, default=NUM_EPOCHS_FPF, help="Number of epochs to train fog pass filter already")

    # Loss weights
    parser.add_argument("--weight-box", type=float, default=WEIGHT_BOX, help="Box loss weight")
    parser.add_argument("--weight-cls", type=float, default=WEIGHT_CLS, help="Classification loss weight")
    parser.add_argument("--weight-obj", type=float, default=WEIGHT_OBJ, help="Object loss weight")
    parser.add_argument("--weight-dfl", type=float, default=WEIGHT_DFL, help="DFL loss weight")
    parser.add_argument("--weight-fsm", type=float, default=WEIGHT_FSM, help="FSM loss weight")
    parser.add_argument("--weight-con", type=float, default=WEIGHT_CON, help="Consistency loss weight")
    return parser.parse_args()

args = get_arguments()
