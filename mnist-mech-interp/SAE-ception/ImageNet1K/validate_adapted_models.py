import torch

# --- Paths to the two models you just saved ---
MODEL_PATH_A = './SAE-Results/results/F0/best_model_lf_0.01.pth'
MODEL_PATH_B = './SAE-Results/results/baseline/baseline_classifier.pth'

# Load the state dictionaries
state_dict_A = torch.load(MODEL_PATH_A, map_location='cpu')
state_dict_B = torch.load(MODEL_PATH_B, map_location='cpu')

print("--- Comparing weights of the two models trained for one batch ---\n")

# Parameters we expect to be DIFFERENT
params_to_check_diff = [
    'head.norm.weight',
    'head.norm.bias'
]

# Parameters we expect to be IDENTICAL (after only one batch)
params_to_check_same = [
    'head.fc.weight',
    'head.fc.bias'
]

all_tests_passed = True

print("--- Checking parameters expected to be DIFFERENT ---")
for param_name in params_to_check_diff:
    tensor_A = state_dict_A[param_name]
    tensor_B = state_dict_B[param_name]
    
    if not torch.equal(tensor_A, tensor_B):
        print(f"✅ PASSED: '{param_name}' weights are DIFFERENT as expected.")
    else:
        print(f"❌ FAILED: '{param_name}' weights are IDENTICAL.")
        all_tests_passed = False

print("\n--- Checking parameters expected to be IDENTICAL ---")
for param_name in params_to_check_same:
    tensor_A = state_dict_A[param_name]
    tensor_B = state_dict_B[param_name]
    
    if torch.equal(tensor_A, tensor_B):
        print(f"✅ PASSED: '{param_name}' weights are IDENTICAL as expected.")
    else:
        print(f"❌ FAILED: '{param_name}' weights are DIFFERENT.")
        all_tests_passed = False

print("\n--- Test Summary ---")
if all_tests_passed:
    print("🎉 Success! The loss_factor is correctly influencing the gradients and weights.")
else:
    print("🤔 Something is still not right in the training dynamics.")