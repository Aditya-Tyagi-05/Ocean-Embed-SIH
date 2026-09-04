"""Train only the spatiotemporal ViT across all monthly dataset folders."""
import sys
from train_models import main
if __name__ == "__main__": sys.argv.extend(["--model", "vit"]); main()
