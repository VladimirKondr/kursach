#!/bin/bash

# Example bash script for running REINVENT with different configurations
# This script demonstrates how to use REINVENT from the command line

# Set variables
PROJECT_DIR="/home/vladimirkondratyonok/projects/kursach"
SMILES_FILE="path/to/smiles.txt"
PRIOR_MODEL="my_prior.pt"
TRAINED_MODEL="my_trained.pt"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Step 1: Create a new prior model
# ==============================================================================

echo -e "${BLUE}Step 1: Creating a new REINVENT prior model${NC}"

# Create output directory
mkdir -p output

# Run model creation
python -m reinvent examples/config_create_model.toml \
    --config-format toml \
    --device cuda

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Prior model created successfully${NC}"
else
    echo "Error creating prior model"
    exit 1
fi

# ==============================================================================
# Step 2: Train the model on custom data (Transfer Learning)
# ==============================================================================

echo -e "${BLUE}Step 2: Training the model via Transfer Learning${NC}"

python -m reinvent examples/config_transfer_learning.toml \
    --config-format toml \
    --device cuda

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model training completed${NC}"
else
    echo "Error during training"
    exit 1
fi

# View training logs with TensorBoard (optional)
# tensorboard --logdir=./logs/transfer_learning

# ==============================================================================
# Step 3: Generate molecules using the trained model
# ==============================================================================

echo -e "${BLUE}Step 3: Generating molecules from trained model${NC}"

python -m reinvent examples/config_sampling.toml \
    --config-format toml \
    --device cuda

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Molecules generated successfully${NC}"
else
    echo "Error during sampling"
    exit 1
fi

# ==============================================================================
# Summary
# ==============================================================================

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All steps completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output files:"
echo "  - Prior model: output/my_prior.pt"
echo "  - Trained model: output/my_trained.pt"
echo "  - Generated molecules: output/generated_molecules.csv"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir=./logs/transfer_learning"
