# How to Get Access to FLUX.1-dev

FLUX.1-dev is a gated model that requires approval from Black Forest Labs. Here's how to get access:

## 🔓 Step-by-Step Access Instructions

### 1. Create HuggingFace Account (if you don't have one)
- Go to https://huggingface.co/join
- Sign up with your email

### 2. Request Access to FLUX.1-dev
1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-dev
2. Click the **"Request Access"** button (orange button on the right)
3. Read and accept the terms and conditions
4. Click **"Submit"**
5. Wait for approval (usually instant, sometimes takes a few hours)

### 3. Get Your HuggingFace Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Choose **"Read"** permissions (sufficient for downloading models)
4. Copy the token (starts with `hf_...`)

### 4. Authenticate in Your Environment

**Method 1: Using CLI (Recommended)**
```bash
huggingface-cli login
# Paste your token when prompted
```

**Method 2: Using Python**
```bash
python -c "from huggingface_hub import login; login()"
# Paste your token when prompted
```

**Method 3: Using Environment Variable**
```bash
export HF_TOKEN="hf_your_token_here"
```

Or add to your `.env` file:
```
HF_TOKEN=hf_your_token_here
```

### 5. Verify Access

Test that you can access the model:

```python
from huggingface_hub import hf_hub_download

try:
    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="README.md"
    )
    print("✅ Access granted! You can now use FLUX.1-dev")
except Exception as e:
    print(f"❌ Access denied: {e}")
```

### 6. Run Training

Once authenticated, you can run training:

```bash
cd RepText
accelerate launch train_arabic.py --config train_config_48gb.yaml --use_wandb
```

---

## ⚡ Alternative: Use SDXL (No Authentication Needed)

If you want to start training immediately without waiting for FLUX approval:

### Option 1: Use Stable Diffusion XL
```bash
accelerate launch train_arabic.py --config train_config_sdxl.yaml --use_wandb
```

**Pros:**
- ✅ No authentication required
- ✅ Start training immediately
- ✅ Still produces high-quality results

**Cons:**
- ❌ Slightly lower quality than FLUX.1-dev
- ❌ Different architecture (ControlNet implementation may differ)

### Option 2: Wait for FLUX Access
```bash
# After getting access to FLUX.1-dev
accelerate launch train_arabic.py --config train_config_48gb.yaml --use_wandb
```

**Pros:**
- ✅ State-of-the-art quality
- ✅ Better text rendering
- ✅ Aligned with RepText paper

**Cons:**
- ❌ Requires approval (may take hours)

---

## 🔍 Troubleshooting

### "401 Unauthorized" Error
- You haven't requested access yet → Go to step 2
- Access not approved yet → Wait a few hours and try again
- Not authenticated → Go to step 4

### "403 Forbidden" Error
- Your token doesn't have the right permissions → Create a new token with "Read" access

### Token Not Working
```bash
# Clear cached credentials
rm -rf ~/.huggingface/token

# Re-authenticate
huggingface-cli login
```

### Check Your Authentication Status
```bash
huggingface-cli whoami
```

This should show your username if you're logged in.

---

## 📞 Still Having Issues?

1. Check if you accepted the FLUX.1-dev license agreement
2. Verify your token starts with `hf_`
3. Make sure you're using the same account for requesting access and the token
4. Try logging out and logging back in:
   ```bash
   huggingface-cli logout
   huggingface-cli login
   ```

5. Use SDXL as a temporary alternative while troubleshooting FLUX access

---

## 📚 Resources

- [FLUX.1-dev Model Card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [HuggingFace Authentication Docs](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
- [HuggingFace Tokens](https://huggingface.co/settings/tokens)
