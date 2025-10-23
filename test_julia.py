from compute import compute_grid
arr = compute_grid('julia', -1.5, 1.5, -1.0, 1.0, 400, 300, 256, c0=complex(-0.4, 0.6))
print("shape:", arr.shape)
print("min, max:", arr.min(), arr.max())
print("dtype:", arr.dtype)
# quick sanity: some values should be >0 and <= max_iters
print("some sample values:", arr[0,0], arr[arr.shape[0]//2, arr.shape[1]//2])
