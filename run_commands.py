import subprocess
import time
import sys



def main(package_name):
    for j in range(1, 23): # Assuming there are 22 queries
        for i in range(1, 16):  # Run the command 15 times
            command = f"{sys.executable} -m {package_name}.q{j}"
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <package_name>")
    else:
        main(sys.argv[1])