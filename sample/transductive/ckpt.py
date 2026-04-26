import glob
import re

files = glob.glob("./data/nell/saveModel/*.pt")

best_file = None
best_val = -1
best_test = 0

for f in files:
    match = re.search(r"ValMRR_([\d.]+)_TestMRR_([\d.]+)\.pt", f)

    if not match:
        continue

    val = float(match.group(1))
    test = float(match.group(2))

    print(f"Checking: VAL={val} TEST={test}")

    if val > best_val:
        best_val = val
        best_test = test
        best_file = f

print("\n====================")
print(f"[VALID] MRR:{best_val:.4f} H@1:N/A H@10:N/A")
print(f"[TEST]  MRR:{best_test:.4f} H@1:N/A H@10:N/A")
print(f"[TIME]  train:N/A inference:N/A")

print("\nBEST CHECKPOINT:")
print(best_file)
