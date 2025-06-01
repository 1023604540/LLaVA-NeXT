from llava.train.train import train
import os
os.environ["DECORD_FFMPEG_LOG_LEVEL"] = "error"

if __name__ == "__main__":
    train()
