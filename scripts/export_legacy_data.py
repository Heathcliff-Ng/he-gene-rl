#!/usr/bin/env python3
"""Export the historical Rank cohorts into the portable image-bank format.

This optional migration tool reads legacy files directly and imports no Rank
modules. It verifies frozen hashes and cell/gene/embedding joins before export.
It does not train, predict, evaluate or modify the source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


PARTS = ('train', 'policy', 'monitor', 'calibration', 'test')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(value, root):
    """Rebase paths from the original machine onto the explicit Rank root."""
    path = Path(value)
    if not path.is_absolute():
        result = root / path
    else:
        try:
            result = root / path.relative_to(root)
        except ValueError:
            indices = [i for i, component in enumerate(path.parts)
                       if component in ('outputs', 'datasets')]
            if not indices:
                raise ValueError(f'Cannot rebase legacy path under --rank-root: {value}')
            result = root.joinpath(*path.parts[indices[0]:])
    result = result.resolve()
    if root not in result.parents:
        raise ValueError(f'Source path escapes --rank-root: {value}')
    return result


def verify(path, expected, provenance, root):
    actual = sha256(path)
    if not expected or actual != expected:
        raise ValueError(f'Frozen legacy input checksum mismatch: {path}')
    provenance[path.relative_to(root).as_posix()] = actual
    return path


def save_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise FileExistsError(f'Existing export has different content: {path}')
        return
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def save_npz(path, **arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                    not np.array_equal(existing[key], value) for key, value in arrays.items()):
                raise FileExistsError(f'Existing export has different content: {path}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        # Link creation fails instead of overwriting an existing destination.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_bank(source, target, mode):
    expected = sha256(source)
    if target.exists():
        if sha256(target) != expected:
            raise FileExistsError(f'Existing bank differs from source: {target}')
        return
    if mode == 'hardlink':
        os.link(source, target)
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                with source.open('rb') as source_stream:
                    shutil.copyfileobj(source_stream, stream, length=8 * 1024 * 1024)
            if sha256(temporary) != expected:
                raise ValueError(f'Copied bank checksum mismatch: {source}')
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def export(rank_root, output, sections=None, bank_mode='copy'):
    root, output = Path(rank_root).resolve(), Path(output).resolve()
    legacy = root / 'outputs/masked_count1_20261020'
    audit_path, config_path = legacy / 'data/audit.json', legacy / 'config.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    expected = {source_path(path, root): digest for path, digest in audit['identity']['inputs'].items()}
    sections = list(sections or audit['identity']['sections'])
    if not sections or len(set(sections)) != len(sections):
        raise ValueError('Choose at least one unique section')
    if bank_mode not in ('copy', 'hardlink'):
        raise ValueError('bank_mode must be copy or hardlink')
    output.mkdir(parents=True, exist_ok=True)
    genes = np.asarray(audit['gene_ids']).astype(str)
    if genes.ndim != 1 or not len(genes) or len(np.unique(genes)) != len(genes):
        raise ValueError('Legacy gene identities must be a unique vector')
    manifest = {'format': 'he-gene-rl-data-v1', 'sections': {}, 'checksums': {}}
    provenance = {audit_path.relative_to(root).as_posix(): sha256(audit_path),
                  config_path.relative_to(root).as_posix(): sha256(config_path)}
    for section in sections:
        if (not isinstance(section, str) or not section
                or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in section)):
            raise ValueError('Section names must use letters, digits, underscores or hyphens')
        if section not in audit['sources']:
            raise ValueError(f'Section absent from legacy audit: {section}')
        bank_source = source_path(audit['sources'][section]['bank_features'], root)
        metadata_source = source_path(audit['sources'][section]['bank_metadata'], root)
        verify(bank_source, expected.get(bank_source), provenance, root)
        verify(metadata_source, expected.get(metadata_source), provenance, root)
        bank = np.load(bank_source, mmap_mode='r', allow_pickle=False)
        with np.load(metadata_source, allow_pickle=False) as archive:
            bank_ids = archive['cell_ids'].astype(str)
            bank_genes = archive['gene_ids'].astype(str)
            bank_coordinates = archive['image_coordinates']
        if (bank.ndim != 2 or bank.dtype.kind not in 'fiu' or bank.shape[1] < 1
                or bank_ids.ndim != 1 or len(bank_ids) != len(bank)
                or len(np.unique(bank_ids)) != len(bank_ids)
                or bank_coordinates.shape != (len(bank), 2)
                or not np.isfinite(bank_coordinates).all()
                or not np.array_equal(bank_genes, genes)):
            raise ValueError(f'Legacy bank identity/shape failure: {section}')
        for start in range(0, len(bank), 4096):
            if not np.isfinite(bank[start:start + 4096]).all():
                raise ValueError(f'Nonfinite legacy bank features: {section}')
        target_dir = output / section
        target_dir.mkdir(exist_ok=True)
        export_bank(bank_source, target_dir / 'features.npy', bank_mode)
        save_npz(target_dir / 'metadata.npz', cell_ids=bank_ids, gene_ids=genes,
                 image_coordinates=bank_coordinates)
        section_spec = {'features': f'{section}/features.npy',
                        'metadata': f'{section}/metadata.npz', 'splits': {}}
        seen_ids = set()
        for part in PARTS:
            if part == 'policy':
                source = legacy / 'data' / section / 'policy.npz'
                digest = audit['cache_sha256'][f'{section}/policy.npz']
            else:
                source = source_path(config['data']['cohort_root'], root) / 'data' / section / f'{part}.npz'
                digest = expected.get(source)
            verify(source, digest, provenance, root)
            with np.load(source, allow_pickle=False) as archive:
                rows = archive['row_indices']
                ids = archive['cell_ids'].astype(str)
                counts = archive['counts']
                center = archive['features']
                coordinates = archive['coordinates']
                tiles = archive['tiles'] if 'tiles' in archive.files else None
            if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
                    or np.any(rows < 0) or np.any(rows >= len(bank))
                    or len(np.unique(rows)) != len(rows)
                    or ids.shape != rows.shape or len(np.unique(ids)) != len(ids)
                    or not np.array_equal(bank_ids[rows], ids)
                    or not np.array_equal(bank_coordinates[rows], coordinates)):
                raise ValueError(f'Legacy cell/row/coordinate identity failure: {section}/{part}')
            if set(ids) & seen_ids:
                raise ValueError(f'Legacy source split identities overlap: {section}/{part}')
            seen_ids.update(ids)
            if center.shape != (len(rows), bank.shape[1]):
                raise ValueError(f'Legacy center feature shape failure: {section}/{part}')
            for start in range(0, len(rows), 4096):
                if not np.array_equal(bank[rows[start:start + 4096]], center[start:start + 4096]):
                    raise ValueError(f'Legacy center features differ from bank: {section}/{part}')
            if (counts.shape != (len(rows), len(genes)) or counts.dtype.kind not in 'fiu'
                    or not np.isfinite(counts).all() or np.any(counts < 0)
                    or np.any(counts != np.floor(counts))):
                raise ValueError(f'Invalid legacy raw counts: {section}/{part}')
            arrays = {'row_indices': rows, 'cell_ids': ids, 'gene_ids': genes,
                      'counts': counts, 'coordinates': coordinates}
            if tiles is not None:
                if (tiles.ndim not in (1, 2) or len(tiles) != len(rows)
                        or (tiles.ndim == 2 and tiles.shape[1] < 1)
                        or tiles.dtype.kind not in 'USfiu'
                        or (tiles.dtype.kind in 'fiu' and not np.isfinite(tiles).all())):
                    raise ValueError(f'Invalid legacy spatial blocks: {section}/{part}')
                arrays['tiles'] = tiles
            elif part != 'policy':
                raise ValueError(f'Missing historical spatial blocks: {section}/{part}')
            save_npz(target_dir / f'{part}.npz', **arrays)
            section_spec['splits'][part] = f'{section}/{part}.npz'
        manifest['sections'][section] = section_spec
        for relative in [section_spec['features'], section_spec['metadata'], *section_spec['splits'].values()]:
            manifest['checksums'][relative] = sha256(output / relative)
    save_json(output / 'export_provenance.json', {
        'format': 'he-gene-rl-legacy-export-v1', 'source_layout': 'Rank',
        'source_files_sha256': provenance, 'sections': sections,
        'bank_mode': bank_mode, 'source_modified': False,
        'split_membership_and_order_preserved': True,
        'note': 'Hardlinked banks share storage with their source and must remain read-only.'
                if bank_mode == 'hardlink' else 'Image banks are independent copies.'})
    save_json(output / 'manifest.json', manifest)
    return output / 'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank-root', type=Path, required=True,
                        help='Path to the existing Rank checkout containing frozen outputs')
    parser.add_argument('--output', type=Path, required=True,
                        help='Destination directory for portable data and manifest.json')
    parser.add_argument('--sections', nargs='+', help='Default: all sections in the original audit')
    parser.add_argument('--bank-mode', choices=('copy', 'hardlink'), default='copy',
                        help='hardlink saves storage but shares the original bank inode')
    args = parser.parse_args()
    result = export(args.rank_root, args.output, args.sections, args.bank_mode)
    print(f'Exported portable data manifest: {result}')


if __name__ == '__main__':
    main()
