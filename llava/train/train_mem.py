from llava.train.train import train
import os
os.environ["FFMPEG_LOG_LEVEL"] = "fatal"
os.environ["DECORD_FFMPEG_LOG_LEVEL"] = "fatal"
if __name__ == "__main__":
    train()
