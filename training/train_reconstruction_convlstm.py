import sys
from train_reconstruction import main
if __name__=='__main__':sys.argv.extend(['--model','convlstm']);main()
