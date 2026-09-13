#!/bin/bash
# ============================================================
# deploy_to_server.sh
# Transfers code (and optional preprocessed test data) to server.
#
# Usage:
#   ./scripts/deploy_to_server.sh USER@SERVER PORT [--with-test-data]
#
# Example:
#   ./scripts/deploy_to_server.sh mahdi@server.shahed.ac.ir 22 --with-test-data
# ============================================================

set -euo pipefail

# ---- Args ----
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 USER@SERVER PORT [--with-test-data]"
    echo "Example: $0 mahdi@server.shahed.ac.ir 22 --with-test-data"
    exit 1
fi

REMOTE="$1"
PORT="$2"
WITH_TEST_DATA="${3:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "📁 Project root: $PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CODE_TAR="deploy_code_${TIMESTAMP}.tar.gz"
TEST_TAR="deploy_testdata_${TIMESTAMP}.tar.gz"

# ---- 1. Package code (exclude data) ----
echo "📦 Packaging code..."
tar -czf "$CODE_TAR" \
    --exclude='./data' \
    --exclude='./work_dir' \
    --exclude='./results' \
    --exclude='./.git' \
    --exclude='./__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.tar.gz' \
    -C "$PROJECT_ROOT" .

echo "   → $CODE_TAR ($(du -h "$CODE_TAR" | cut -f1))"

# ---- 2. Package preprocessed test data (optional) ----
if [ "$WITH_TEST_DATA" = "--with-test-data" ]; then
    if [ -d "./data/brain_test" ]; then
        echo "📦 Packaging preprocessed test data..."
        tar -czf "$TEST_TAR" -C "$PROJECT_ROOT/data" brain_test
        echo "   → $TEST_TAR ($(du -h "$TEST_TAR" | cut -f1))"
    else
        echo "⚠️  ./data/brain_test not found. Run split_dataset.py first."
        echo "   Skipping test data packaging."
        WITH_TEST_DATA=""
    fi
fi

# ---- 3. Create remote dir and upload ----
echo ""
echo "🚀 Uploading to $REMOTE ..."

ssh -p "$PORT" "$REMOTE" "mkdir -p ~/SAM-Med3D"
echo "   Uploading code..."
scp -P "$PORT" "$CODE_TAR" "$REMOTE:~/SAM-Med3D/"

if [ "$WITH_TEST_DATA" = "--with-test-data" ]; then
    echo "   Uploading test data..."
    scp -P "$PORT" "$TEST_TAR" "$REMOTE:~/SAM-Med3D/"
fi

# ---- 4. Remote extraction ----
echo ""
echo "📂 Extracting on server..."
ssh -p "$PORT" "$REMOTE" "cd ~/SAM-Med3D && \
    tar -xzf $CODE_TAR && \
    rm -f $CODE_TAR && \
    mkdir -p data results scripts && \
    echo '   Code extracted.'"

if [ "$WITH_TEST_DATA" = "--with-test-data" ]; then
    ssh -p "$PORT" "$REMOTE" "cd ~/SAM-Med3D && \
        mkdir -p data && \
        tar -xzf $TEST_TAR -C data/ && \
        rm -f $TEST_TAR && \
        echo '   Test data extracted to data/brain_test/'"
fi

# ---- 5. Cleanup local tarballs ----
rm -f "$CODE_TAR"
[ -f "$TEST_TAR" ] && rm -f "$TEST_TAR"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps on the server:"
echo "  ssh -p $PORT $REMOTE"
echo "  conda create --name sammed3d_gpu python=3.10 -y"
echo "  conda activate sammed3d_gpu"
echo "  nvidia-smi"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo "  pip install uv && uv pip install torchio monai edt surface-distance medim nibabel pandas scipy tqdm matplotlib"
echo "  cd ~/SAM-Med3D"
echo "  # Download full dataset (if training):"
echo "  wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
echo "  tar -xvf Task01_BrainTumour.tar && rm Task01_BrainTumour.tar"
