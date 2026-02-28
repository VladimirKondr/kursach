"""
Practical Examples for REINVENT Model Usage

This script demonstrates how to use the REINVENT model programmatically
for:
1. Creating a new prior model
2. Training via Transfer Learning
3. Generating molecules (inference)
4. Computing likelihood of molecules
"""

import torch
import logging
from typing import List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ==============================================================================
# EXAMPLE 1: Loading and using a pre-trained REINVENT model
# ==============================================================================

def example_load_model(model_path: str):
    """
    Load a pre-trained REINVENT model and inspect it.
    
    Args:
        model_path: Path to the saved model file (.pt)
    
    Returns:
        Loaded model object
    """
    from reinvent.models.reinvent.models.model import Model
    
    logger.info(f"Loading model from {model_path}")
    
    # Determine device (GPU if available, CPU otherwise)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model from checkpoint
    model = Model.create_from_dict(
        checkpoint, 
        mode="inference",  # Use inference mode for sampling
        device=device
    )
    
    # Print model information
    logger.info(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
    logger.info(f"Model version: {checkpoint.get('version', 'Unknown')}")
    logger.info(f"Max sequence length: {model.max_sequence_length}")
    logger.info(f"Vocabulary size: {len(model.vocabulary)}")
    logger.info(f"Vocabulary tokens: {len(model.vocabulary.tokens())}")
    
    # Print vocabulary (first 20 tokens)
    tokens = model.vocabulary.tokens()
    logger.debug(f"First 20 tokens: {tokens[:20]}")
    
    return model


# ==============================================================================
# EXAMPLE 2: Computing likelihood of molecules
# ==============================================================================

def example_compute_likelihood(model, smiles_list: List[str]) -> List[float]:
    """
    Compute the likelihood (negative log likelihood) of SMILES strings
    according to the model.
    
    Lower NLL = model thinks this is a more probable molecule
    Higher NLL = model thinks this is less probable
    
    Args:
        model: REINVENT model object
        smiles_list: List of SMILES strings
    
    Returns:
        List of NLL values
    """
    logger.info(f"Computing likelihood for {len(smiles_list)} molecules")
    
    nlls = []
    
    with torch.no_grad():
        for smiles in smiles_list:
            try:
                # Compute likelihood for this SMILES
                nll = model.likelihood_smiles([smiles])
                nll_value = nll.item()
                nlls.append(nll_value)
                logger.debug(f"SMILES: {smiles:50s} NLL: {nll_value:8.4f}")
            except Exception as e:
                logger.warning(f"Error processing {smiles}: {e}")  # noqa: F841
                nlls.append(float('inf'))
    
    return nlls


# ==============================================================================
# EXAMPLE 3: Generating molecules (sampling)
# ==============================================================================

def example_generate_molecules(model, num_molecules: int = 100) -> List[str]:
    """
    Generate molecules from the model using random sampling.
    
    Args:
        model: REINVENT model object
        num_molecules: Number of molecules to generate
    
    Returns:
        List of generated SMILES strings
    """
    from reinvent.runmodes.samplers.reinvent import ReinventSampler
    
    logger.info(f"Generating {num_molecules} molecules")
    
    # Create sampler
    sampler = ReinventSampler(model)
    
    generated_smiles = []
    generated_nlls = []
    
    with torch.no_grad():
        # Sample molecules
        for _ in range(num_molecules):
            # Sample one molecule
            sample_batch = sampler.sample(num_samples=1)
            
            if sample_batch.smilies:
                smiles = sample_batch.smilies[0]
                nlls = sample_batch.nlls[0] if sample_batch.nlls else 0.0
                generated_smiles.append(smiles)
                generated_nlls.append(nlls)
    
    logger.info(f"Generated {len(generated_smiles)} valid molecules")
    return generated_smiles


# ==============================================================================
# EXAMPLE 4: Batch generation with statistics
# ==============================================================================

def example_batch_sampling(model, num_molecules: int = 1000) -> Tuple[List[str], dict]:
    """
    Generate molecules in batches and collect statistics.
    
    Args:
        model: REINVENT model object
        num_molecules: Total number of molecules to generate
    
    Returns:
        Tuple of (generated SMILES, statistics dictionary)
    """
    from reinvent.runmodes.setup_sampler import setup_sampler
    
    logger.info(f"Batch sampling {num_molecules} molecules")
    
    # Setup sampler
    params = {
        "batch_size": 100,  # Process 100 at a time
        "randomize_smiles": True,
        "temperature": 1.0,
    }
    sampler, _ = setup_sampler("Reinvent", params, model)
    sampler.unique_sequences = False
    
    generated_smiles = []
    nlls = []
    valid_count = 0
    
    with torch.no_grad():
        # Generate in batches
        num_batches = (num_molecules + 99) // 100
        for batch_idx in range(num_batches):
            batch_size = min(100, num_molecules - len(generated_smiles))
            
            # Sample batch
            sample_batch = sampler.sample(num_samples=batch_size)
            
            if hasattr(sample_batch, 'smilies'):
                for smiles, state, nll in zip(
                    sample_batch.smilies,
                    sample_batch.smilies_state,
                    sample_batch.nlls
                ):
                    generated_smiles.append(smiles)
                    nlls.append(nll)
                    if state == "valid":
                        valid_count += 1
            
            logger.info(f"Batch {batch_idx + 1}/{num_batches}: Generated {len(generated_smiles)} molecules")
    
    # Compute statistics
    stats = {
        "total_generated": len(generated_smiles),
        "valid_molecules": valid_count,
        "validity_rate": valid_count / len(generated_smiles) if generated_smiles else 0.0,
        "mean_nll": float(np.mean(nlls)) if nlls else 0.0,
        "std_nll": float(np.std(nlls)) if nlls else 0.0,
        "min_nll": float(np.min(nlls)) if nlls else 0.0,
        "max_nll": float(np.max(nlls)) if nlls else 0.0,
    }
    
    logger.info("Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return generated_smiles, stats


# ==============================================================================
# EXAMPLE 5: Filtering valid molecules
# ==============================================================================

def example_filter_valid_molecules(model, smiles_list: List[str]) -> Tuple[List[str], dict]:
    """
    Filter molecules that are valid SMILES according to RDKit.
    
    Args:
        model: REINVENT model object
        smiles_list: List of SMILES strings to validate
    
    Returns:
        Tuple of (valid SMILES, statistics)
    """
    from rdkit import Chem
    
    logger.info(f"Filtering {len(smiles_list)} molecules for validity")
    
    valid_smiles = []
    valid_nlls = []
    invalid_count = 0
    
    # Compute likelihoods
    nlls = example_compute_likelihood(model, smiles_list)
    
    with torch.no_grad():
        for smiles, nll in zip(smiles_list, nlls):
            try:
                # Check if SMILES is valid
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
                    valid_nlls.append(nll)
                else:
                    invalid_count += 1
            except Exception:  # noqa: F841
                invalid_count += 1
    
    stats = {
        "total": len(smiles_list),
        "valid": len(valid_smiles),
        "invalid": invalid_count,
        "validity_rate": len(valid_smiles) / len(smiles_list) if smiles_list else 0.0,
    }
    
    logger.info(f"Filtering results: {stats['valid']}/{stats['total']} valid molecules")
    
    return valid_smiles, stats


# ==============================================================================
# EXAMPLE 6: Computing molecular properties for generated molecules
# ==============================================================================

def example_molecular_properties(smiles_list: List[str]) -> dict:
    """
    Compute molecular properties for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        Dictionary with property statistics
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    
    logger.info(f"Computing properties for {len(smiles_list)} molecules")
    
    mw_list = []
    logp_list = []
    hbd_list = []
    hba_list = []
    aromatic_rings = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw_list.append(Descriptors.MolWt(mol))
                logp_list.append(Crippen.MolLogP(mol))
                hbd_list.append(Descriptors.NumHDonors(mol))
                hba_list.append(Descriptors.NumHAcceptors(mol))
                aromatic_rings.append(Descriptors.NumAromaticRings(mol))
        except Exception:  # noqa: F841
            pass
    
    properties = {
        "molecular_weight": {
            "mean": float(np.mean(mw_list)),
            "std": float(np.std(mw_list)),
            "min": float(np.min(mw_list)),
            "max": float(np.max(mw_list)),
        },
        "logp": {
            "mean": float(np.mean(logp_list)),
            "std": float(np.std(logp_list)),
            "min": float(np.min(logp_list)),
            "max": float(np.max(logp_list)),
        },
        "h_donors": {
            "mean": float(np.mean(hbd_list)),
            "max": float(np.max(hbd_list)),
        },
        "h_acceptors": {
            "mean": float(np.mean(hba_list)),
            "max": float(np.max(hba_list)),
        },
        "aromatic_rings": {
            "mean": float(np.mean(aromatic_rings)),
            "max": float(np.max(aromatic_rings)),
        },
    }
    
    logger.info("Molecular Properties:")
    for prop, stats in properties.items():
        logger.info(f"  {prop}: {stats}")
    
    return properties


# ==============================================================================
# EXAMPLE 7: Complete workflow
# ==============================================================================

def example_complete_workflow(model_path: str, output_file: str = "generated_molecules.csv"):
    """
    Complete workflow: Load model -> Generate molecules -> Filter -> Analyze
    
    Args:
        model_path: Path to the trained model
        output_file: Output file for generated molecules
    """
    import csv
    
    logger.info("=" * 80)
    logger.info("STARTING COMPLETE WORKFLOW")
    logger.info("=" * 80)
    
    # Step 1: Load model
    model = example_load_model(model_path)
    logger.info("✓ Model loaded successfully")
    
    # Step 2: Generate molecules
    generated_smiles, gen_stats = example_batch_sampling(model, num_molecules=1000)
    logger.info("✓ Molecules generated")
    
    # Step 3: Filter valid molecules
    valid_smiles, filter_stats = example_filter_valid_molecules(model, generated_smiles)
    logger.info(f"✓ Molecules filtered: {filter_stats['valid']}/{filter_stats['total']} valid")
    
    # Step 4: Compute likelihoods for valid molecules
    nlls = example_compute_likelihood(model, valid_smiles)
    logger.info("✓ Likelihoods computed")
    
    # Step 5: Compute molecular properties
    properties = example_molecular_properties(valid_smiles)
    logger.info("✓ Molecular properties computed")
    
    # Step 6: Save results to CSV
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES', 'NLL', 'Molecular_Weight', 'LogP', 'H_Donors', 'H_Acceptors'])
        
        for smiles, nll in zip(valid_smiles, nlls):
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, Crippen
                
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mw = Descriptors.MolWt(mol)
                    logp = Crippen.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    writer.writerow([smiles, f"{nll:.4f}", f"{mw:.2f}", f"{logp:.2f}", hbd, hba])
            except Exception:  # noqa: F841
                pass
    
    logger.info("=" * 80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Generated {filter_stats['valid']} valid molecules")
    logger.info(f"Results saved to {output_file}")
    
    return valid_smiles, nlls, properties


# ==============================================================================
# Test examples
# ==============================================================================

if __name__ == "__main__":
    """
    Run example code. Make sure you have:
    1. A trained REINVENT model (.pt file)
    2. The REINVENT package installed
    """
    
    # Example SMILES for testing
    test_smiles = [
        "CCO",                                    # ethanol
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O",       # ibuprofen
        "c1ccc(cc1)C(=O)O",                      # benzoic acid
        "CC(=O)Oc1ccccc1C(=O)O",                 # acetylsalicylic acid (aspirin)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",          # caffeine
    ]
    
    print("\n" + "=" * 80)
    print("REINVENT Model Usage Examples")
    print("=" * 80)
    
    print("\nExample 1: Testing SMILES likelihood computation")
    print("Note: These are example SMILES strings. In practice, use a trained model.")
    print("\nExample SMILES:")
    for smiles in test_smiles:
        print(f"  - {smiles}")
    
    print("\n" + "=" * 80)
    print("To run these examples with a real model:")
    print("=" * 80)
    print("""
    from examples import *
    
    # Load your trained model
    model = example_load_model("path/to/your/model.pt")
    
    # Generate molecules
    generated = example_generate_molecules(model, num_molecules=100)
    
    # Compute likelihood
    nlls = example_compute_likelihood(model, generated)
    
    # Complete workflow
    smiles, nlls, props = example_complete_workflow("path/to/your/model.pt")
    """)
