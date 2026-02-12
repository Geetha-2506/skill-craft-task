import subprocess
import sys

scripts = [
    "house_price.py",
    "customer_segmentation.py",
    "svm_cats_dogs.py",
    "hand_gesture_recognition.py"
]

for script in scripts:
    print(f"\nRunning {script} ...")
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"Error running {script}")
        break
    else:
        print(f"{script} completed successfully")

print("\nAll tasks executed.")
