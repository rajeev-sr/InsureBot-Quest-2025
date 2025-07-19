import subprocess
import os

directory_path = "/home/rajeev-kumar/Desktop/InsureBot-Quest-2025/src/components/Sample Call Recordings"
output_file="/home/rajeev-kumar/Desktop/InsureBot-Quest-2025/src/components/output/"


def get_slow_audio(directory_path):

    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
            # Build the ffmpeg command
            command = [
                "ffmpeg",
                "-i", full_path,
                "-af", "afftdn,atempo=0.8",
                "-ac", "1",
                "-ar", "16000",
                output_file+filename
            ]

            # Run the command
            subprocess.run(command)

if __name__ == "__main__":
    get_slow_audio(directory_path)
