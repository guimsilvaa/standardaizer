#!/usr/bin/env python3

import os
import re
import gzip
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count
from molvs import Standardizer
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

# =========================
# Silence RDKit warnings
# =========================
RDLogger.DisableLog("rdApp.*")

# =========================
# Globals
# =========================
molvs_standardizer = Standardizer()
ALLOWED_ATOMS = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
LOGFILE = "standardize.log"

# =========================
# Logging
# =========================
def log(msg=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line)
    with open(LOGFILE, "a") as lf:
        lf.write(line + "\n")

# =========================
# CXSMILES cleaning
# =========================
def strip_cxsmiles(smiles):
    return smiles.split("|", 1)[0].strip() if "|" in smiles else smiles.strip()

def strip_cxsmiles_identifier(identifier):
    if not identifier:
        return ""
    identifier = re.sub(r"\|[^|]*\|", "", identifier)
    return " ".join(identifier.split())

# =========================
# Chemistry helpers
# =========================
def contains_only_allowed_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return all(atom.GetSymbol() in ALLOWED_ATOMS for atom in mol.GetAtoms())

def standardize_smiles(smiles):
    try:
        smiles = strip_cxsmiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = molvs_standardizer.standardize(mol)
        smi = Chem.MolToSmiles(mol)

        if "." in smi:
            smi = max(smi.split("."), key=len)

        if not contains_only_allowed_atoms(smi):
            return None

        return smi
    except Exception:
        return None

def get_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.MolToInchiKey(mol)
    except Exception:
        return None

# =========================
# Worker
# =========================
def process_record(record):
    smiles, identifier = record
    std = standardize_smiles(smiles)
    if std is None:
        return None, smiles, identifier, None
    return std, None, strip_cxsmiles_identifier(identifier), None

# =========================
# Input reader
# =========================
def read_smi_stream(filepath):
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            yield parts[0], " ".join(parts[1:])

# =========================
# Core processing
# =========================
def process_file(filepath, delete_input, use_inchikey, chunksize=2000):
    base = os.path.splitext(filepath)[0]
    out_file = f"{base}_standardized.smi"
    fail_file = f"{base}_failed.txt"

    records = list(read_smi_stream(filepath))
    initial_total = len(records)

    seen = set()
    written = failed = duplicated = 0

    with open(out_file, "w") as out, open(fail_file, "w") as fail:
        fail.write("SMILES\tIDENTIFIER\tREASON\n")

        with Pool(processes=max(1, cpu_count() - 2)) as pool:
            for std, raw_smiles, ident, _ in tqdm(
                pool.imap_unordered(process_record, records, chunksize=chunksize),
                total=initial_total,
                desc=os.path.basename(filepath),
                dynamic_ncols=True,
            ):
                if std is None:
                    fail.write(f"{raw_smiles}\t{ident}\tFAILED_STANDARDIZATION\n")
                    failed += 1
                    continue

                key = get_inchikey(std) if use_inchikey else std
                if key is None:
                    fail.write(f"{std}\t{ident}\tFAILED_STANDARDIZATION\n")
                    failed += 1
                    continue

                if key in seen:
                    fail.write(f"{std}\t{ident}\tDUPLICATE\n")
                    duplicated += 1
                    continue

                seen.add(key)
                out.write(f"{std}\t{ident}\n")
                written += 1

    log("")
    log(f"File processed: {filepath}")
    log(f"Initial number of molecules: {initial_total}")
    log(f"Molecules that failed standardization: {failed}")
    log(f"Duplicated molecules: {duplicated}")
    log(f"Final number of molecules in output file: {written}")
    log(f"Standardized molecules written to {out_file}")
    log(f"Failed and duplicated molecules written to {fail_file}")
    log("Thank you for using my script!")
    log("Author: Guilherme M. Silva - Harvard BIDMC")
    log("")

    if delete_input:
        os.remove(filepath)

# =========================
# CLI helpers
# =========================
def parse_selection(selection, files):
    selection = selection.strip().lower()
    if selection == "all":
        return files

    idx = set()
    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            idx.update(range(int(a), int(b) + 1))
        else:
            idx.add(int(part))

    return [files[i - 1] for i in sorted(idx) if 1 <= i <= len(files)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_inchikey", action="store_true",
                        help="Use InChIKey instead of standardized SMILES for deduplication")
    args = parser.parse_args()

    folder = os.getcwd()
    smi_files = sorted(f for f in os.listdir(folder) if f.endswith(".smi"))

    if not smi_files:
        print("No .smi files found.")
        return

    print("\nAvailable .smi files:")
    for i, f in enumerate(smi_files, 1):
        print(f" {i}: {f}")

    choice = input("\nSelect file(s) (e.g. 1,2 | 1-3 | all): ")
    selected = parse_selection(choice, smi_files)

    if not selected:
        print("No valid files selected.")
        return

    delete_input = input("Delete input file after success? [y/N]: ").lower() == "y"

    mode = "InChIKey" if args.use_inchikey else "standardized SMILES"
    log(f"Deduplication mode: {mode}")

    for smi in selected:
        process_file(
            smi,
            delete_input=delete_input,
            use_inchikey=args.use_inchikey
        )

if __name__ == "__main__":
    main()

