import os
import gzip
import zipfile
import py7zr
from molvs import Standardizer
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

molvs_standardizer = Standardizer() # Initialize MolVS Standardizer

def find_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for file in filenames:
            if file.endswith((".smi", ".sdf", ".gz", ".zip", ".7z")):
                files.append(os.path.join(root, file))
    return files

def read_smi_file(file_path):
    smiles_with_ids = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        smiles_with_ids.append((parts[0], ' '.join(parts[1:])))
        elif file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for name in zip_ref.namelist():
                    if name.endswith('.smi'):
                        with zip_ref.open(name) as file:
                            for line in file:
                                line = line.decode().strip()
                                if line:
                                    parts = line.split()
                                    smiles_with_ids.append((parts[0], ' '.join(parts[1:])))
        elif file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, 'r') as archive:
                for name in archive.getnames():
                    if name.endswith('.smi'):
                        with archive.open(name) as file:
                            for line in file:
                                line = line.decode().strip()
                                if line:
                                    parts = line.split()
                                    smiles_with_ids.append((parts[0], ' '.join(parts[1:])))
        else:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        smiles_with_ids.append((parts[0], ' '.join(parts[1:])))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return smiles_with_ids

def read_sdf_file(file_path):
    smiles_with_ids = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as file:
                suppl = Chem.ForwardSDMolSupplier(file)
                for mol in suppl:
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        identifier = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                        smiles_with_ids.append((smiles, identifier))
        elif file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for name in zip_ref.namelist():
                    if name.endswith('.sdf'):
                        with zip_ref.open(name) as file:
                            suppl = Chem.ForwardSDMolSupplier(file)
                            for mol in suppl:
                                if mol is not None:
                                    smiles = Chem.MolToSmiles(mol)
                                    identifier = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                                    smiles_with_ids.append((smiles, identifier))
        elif file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, 'r') as archive:
                for name in archive.getnames():
                    if name.endswith('.sdf'):
                        with archive.open(name) as file:
                            suppl = Chem.ForwardSDMolSupplier(file)
                            for mol in suppl:
                                if mol is not None:
                                    smiles = Chem.MolToSmiles(mol)
                                    identifier = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                                    smiles_with_ids.append((smiles, identifier))
        else:
            suppl = Chem.SDMolSupplier(file_path)
            for mol in suppl:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    identifier = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                    smiles_with_ids.append((smiles, identifier))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return smiles_with_ids

def contains_only_allowed_atoms(smiles):
    allowed_atoms = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True

def standardize_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = molvs_standardizer.standardize(mol)         # Use MolVS Standardizer for standardization 
        standardized_smiles = Chem.MolToSmiles(mol)         # Get standardized SMILES
        fragments = standardized_smiles.split('.')        # Split by '.' and keep the longest fragment
        if len(fragments) > 1:
            standardized_smiles = max(fragments, key=len)
        if not contains_only_allowed_atoms(standardized_smiles):         # Check for allowed atoms
            return None
        return standardized_smiles
    except Exception as e:
        print(f"Error processing molecule {smiles}: {e}")
        return None

def get_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchikey = AllChem.MolToInchiKey(mol)
        return inchikey
    except Exception as e:
        print(f"Error generating InChIKey for {smiles}: {e}")
        return None

def curate_molecules(smiles_with_ids):
    curated_smiles = []
    failed_molecules = []
    duplicate_molecules = []
    failed_count = 0
    seen_inchikeys = set()
    duplicate_count = 0
    for smiles, identifier in tqdm(smiles_with_ids, desc="Standardizing molecules"):
        standardized_smiles = standardize_molecule(smiles)
        if standardized_smiles:
            inchikey = get_inchikey(standardized_smiles)
            if inchikey and inchikey not in seen_inchikeys:
                seen_inchikeys.add(inchikey)
                curated_smiles.append((standardized_smiles, identifier))
            elif not inchikey:
                failed_count += 1
                failed_molecules.append((smiles, identifier, inchikey))
            else:
                duplicate_count += 1
                duplicate_molecules.append((standardized_smiles, identifier, inchikey))
        else:
            failed_count += 1
            failed_molecules.append((smiles, identifier, None))
    return curated_smiles, failed_count, duplicate_count, failed_molecules, duplicate_molecules

def process_files(file_paths):
    for file_path in file_paths:
        if file_path.endswith('.sdf') or any(file_path.endswith(ext) for ext in ['.gz', '.zip', '.7z']):
            smiles_with_ids = read_sdf_file(file_path)
        else:
            smiles_with_ids = read_smi_file(file_path)        
        initial_total = len(smiles_with_ids)
        curated_smiles, failed_count, duplicate_count, failed_molecules, duplicate_molecules = curate_molecules(smiles_with_ids)
        output_file = f"{os.path.splitext(file_path)[0]}_curated.smi"
        failed_output_file = f"{os.path.splitext(file_path)[0]}_failed.txt"
        with open(output_file, 'w') as file:
            for smiles, identifier in curated_smiles:
                file.write(f"{smiles} {identifier}\n")
        with open(failed_output_file, 'w') as file:
            file.write("SMILES\tIdentifier\tInChIKey\n")
            for smiles, identifier, inchikey in failed_molecules + duplicate_molecules:
                file.write(f"{smiles}\t{identifier}\t{inchikey}\n")
        curated_count = len(curated_smiles)
        print(f"\nFile processed: {file_path}")
        print(f"Initial number of molecules: {initial_total}")
        print(f"Molecules that failed standardization: {failed_count}")
        print(f"Duplicated molecules: {duplicate_count}")
        print(f"Final number of molecules in output file: {curated_count}")
        print(f"\nStandardized molecules written to {output_file}")
        print(f"Failed and duplicated molecules written to {failed_output_file}\n")
        print(f" Thank you for using my script!\n Author: Guilherme M. Silva - Harvard BIDMC - guimsilva@gmail.com\n")

if __name__ == "__main__":
    folder = os.path.dirname(os.path.realpath(__file__))
    files = find_files(folder)
    if not files:
        print("No .smi or .sdf files found.")
    else:
        print("\nFound .smi or .sdf files:")
        for i, file in enumerate(files):
            print(f"{i + 1}: {file}")
    chosen_files = input("\nEnter the number(s) of the dataset file(s) you want to process (e.g. 2 OR 1,3): ")
    chosen_indices = [int(x.strip()) - 1 for x in chosen_files.split(",")]
    chosen_file_paths = [files[i] for i in chosen_indices]
    process_files(chosen_file_paths)

