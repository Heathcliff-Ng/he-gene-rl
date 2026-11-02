#!/usr/bin/env python3
"""Copy historical checkpoints and reports without importing ML dependencies.

The source archive remains unchanged. Historical model files are checked
against their original freeze manifests before copying. This is a file import,
not training, inference, checkpoint deserialization, or a runtime validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEACHER = Path('experiments/context_information_20261030/artifacts_full')
POLICY = Path('experiments/predicted_history_rl_20261030/artifacts_full')
ONE_SHOT = Path('experiments/predicted_count_inference_20261030/artifacts_full/comparison.json')
STATUS = 'not_runtime_verified_after_migration'


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def relative_source(value: str) -> str:
    """Remove historical machine roots from exported text and metadata."""
    normalized = value.replace('\\', '/')
    for anchor in ('experiments/', 'outputs/'):
        offset = normalized.find(anchor)
        if offset >= 0 and (offset == 0 or normalized[offset - 1] == '/'):
            return normalized[offset:]
    if normalized.startswith('/') or (len(normalized) > 2 and normalized[1:3] == ':/'):
        return '<external path omitted>'
    return value


def portable(value):
    if isinstance(value, dict):
        return {relative_source(str(key)): portable(item)
                for key, item in value.items() if key != 'input_hashes'}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return relative_source(value)
    return value


def safe_component(value) -> str:
    text = str(value)
    if not text or text in ('.', '..') or '/' in text or '\\' in text:
        raise ValueError(f'Unsafe source or seed name: {text!r}')
    return text


def frozen_digest(frozen: dict, relative: Path) -> str:
    wanted = relative.as_posix()
    matches = set()
    for collection in ('models', 'source_files', 'inputs'):
        for recorded, expected in frozen.get(collection, {}).items():
            if relative_source(recorded) == wanted:
                matches.add(expected)
    if len(matches) != 1:
        raise ValueError(f'Missing or inconsistent original frozen digest: {wanted}')
    return matches.pop()


def compatible_destination(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f'Refusing a symlink destination: {path}')
    if path.exists() and (not path.is_file() or path.read_bytes() != content):
        raise FileExistsError(f'Refusing to replace different existing content: {path}')


def write_new_or_identical(path: Path, content: bytes) -> None:
    compatible_destination(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(content)
    except FileExistsError:
        compatible_destination(path, content)


def build_import(rank_root: Path, output: Path, reports: Path):
    provenance_sources = {}

    def read(relative: Path, frozen=None):
        content = (rank_root / relative).read_bytes()
        actual = digest(content)
        if frozen is not None and actual != frozen_digest(frozen, relative):
            raise ValueError(f'Original frozen file changed: {relative.as_posix()}')
        provenance_sources[relative.as_posix()] = actual
        return content

    teacher_frozen = json.loads(read(TEACHER / 'training_frozen.json'))
    policy_frozen = json.loads(read(POLICY / 'training_frozen.json'))
    teacher_config = teacher_frozen['config']
    config = json.loads(json.dumps(policy_frozen['config']))
    config['pretraining'] = teacher_config['training']
    config['hidden_dim'] = teacher_config.get('hidden_dim', 128)
    config['evaluation']['targets'] = list(config['sources'])
    if config['sources'] != teacher_config['sources'] or config['seeds'] != teacher_config['seeds']:
        raise ValueError('Teacher and PPO source/seed settings differ')
    if config['neighbors'] != teacher_config['neighbors']:
        raise ValueError('Teacher and PPO neighbor settings differ')

    run_files = {}
    copied = []
    summaries = {}
    source_gene_ids = None

    def copy(relative: Path, destination: Path, frozen):
        content = read(relative, frozen)
        key = destination.as_posix()
        run_files[key] = content
        copied.append({'source': relative.as_posix(), 'destination': key,
                       'sha256': digest(content), 'bytes': len(content)})

    for source_value in config['sources']:
        source = safe_component(source_value)
        identity = json.loads(read(TEACHER / source / 'data_identity.json', teacher_frozen))
        genes = identity['gene_ids']
        if (not isinstance(genes, list) or not genes
                or not all(isinstance(gene, str) and gene for gene in genes)
                or len(set(genes)) != len(genes)):
            raise ValueError('Historical gene IDs must be unique nonempty strings')
        if source_gene_ids is not None and source_gene_ids != genes:
            raise ValueError('Historical source gene orders differ')
        source_gene_ids = genes
        run_files[f'{source}/genes.json'] = json_bytes(genes)
        copy(POLICY / source / 'normalization.npz', Path(source) / 'normalization.npz', policy_frozen)
        teacher_normalization = read(TEACHER / source / 'normalization.npz', teacher_frozen)
        if teacher_normalization != run_files[f'{source}/normalization.npz']:
            raise ValueError('Teacher and final source normalization differ')
        copy(POLICY / source / 'split_ids.npz', Path(source) / 'split_ids.npz', policy_frozen)
        for seed_value in config['seeds']:
            if isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
                raise ValueError('Historical seeds must be nonnegative integers')
            seed = safe_component(f'seed{seed_value}')
            relative_run = Path(source) / seed
            stages = [
                ('pretraining', TEACHER / relative_run / 'neighbors', teacher_frozen),
                ('adaptation', POLICY / relative_run / 'adaptation', policy_frozen),
                ('ppo', POLICY / relative_run / 'ppo', policy_frozen),
            ]
            summaries[relative_run.as_posix()] = {}
            for stage, old_directory, frozen in stages:
                copy(old_directory / 'best.pt', relative_run / stage / 'best.pt', frozen)
                summary = portable(json.loads(read(old_directory / 'summary.json', frozen)))
                summaries[relative_run.as_posix()][stage] = summary
                run_files[(relative_run / stage / 'summary.json').as_posix()] = json_bytes(summary)

    run_files['run.json'] = json_bytes({
        'format': 'he-gene-rl-run-v1', 'origin': 'legacy-import',
        'config': config, 'read_only': True,
    })
    run_files['training_summary.json'] = json_bytes(summaries)
    run_files['import_manifest.json'] = json_bytes({
        'format': 'he-gene-rl-legacy-import-v1',
        'copied_files': copied, 'source_files': provenance_sources.copy(),
        'validation_status': STATUS,
        'operation': 'Verified file copying only; no model deserialization, training or inference.',
    })
    artifacts = {name: digest(content) for name, content in sorted(run_files.items())}
    run_files['training_frozen.json'] = json_bytes({
        'format': 'he-gene-rl-freeze-v1', 'origin': 'legacy-import', 'config': config,
        'artifacts': artifacts, 'complete': True, 'validation_status': STATUS,
    })

    evaluation = portable(json.loads(read(POLICY / 'evaluation_summary.json')))
    one_shot = json.loads(read(ONE_SHOT))
    static_directions = {}
    for direction, values in one_shot['directions'].items():
        static_directions[direction] = {
            'means': values['means']['context_static'],
            'comparisons': {key: portable(value) for key, value in values['comparisons'].items()
                            if key.startswith('context_static_vs_')},
        }
    legacy_summary = {
        'format': 'he-gene-rl-legacy-results-v1',
        'origin': 'Historical experiments before the standalone-project migration.',
        'validation_status': STATUS,
        'tests_training_or_inference_executed_during_import': False,
        'evaluation': evaluation,
        'teacher_one_shot': {
            'method': 'context_static',
            'description': 'Pretrained H&E plus neighbor predictor with empty history; select all genes with utility > 0.',
            'source_report': ONE_SHOT.as_posix(),
            'directions': static_directions,
        },
        'interpretation': [
            'These are archived results, not a new evaluation of the reorganized package.',
            'Primary cross-section evaluations use two sections of the same specimen and previously examined tests.',
            'Spatial block intervals do not establish independent-patient generalization.',
            'FP+2FN uses raw0 as negative, raw>=2 as positive and raw1 as neutral.',
            'The historical teacher one-shot costs are lower than the PPO costs in both primary cross-section directions.',
            'PPO versus the same adapted predictor is the reported incremental RL comparison; complete-system baseline differences are not all attributable to PPO.',
        ],
    }
    provenance = {
        'format': 'he-gene-rl-provenance-v1',
        'validation_status': STATUS,
        'path_convention': 'Source paths are relative to the explicitly supplied original Rank checkout.',
        'source_files': dict(sorted(provenance_sources.items())),
        'checkpoint_mapping': copied,
        'notes': [
            'Checkpoint, normalization, split and stage-summary inputs were verified against original frozen SHA256 records.',
            'Summary reports are copied from historical aggregate JSON; no raw RNA or prediction evaluation is read or run.',
            'Machine-specific absolute paths and input_hashes were removed from exported human-readable metadata.',
            'No model was loaded into PyTorch; migration runtime and numerical parity checks remain pending.',
        ],
    }
    files = {output / name: content for name, content in run_files.items()}
    files[reports / 'legacy_summary.json'] = json_bytes(legacy_summary)
    files[reports / 'provenance.json'] = json_bytes(provenance)
    return files, copied


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank-root', type=Path, required=True,
                        help='Original Rank checkout, containing experiments/.')
    parser.add_argument('--output', type=Path, default=project / 'runs/legacy')
    parser.add_argument('--reports-dir', type=Path, default=project / 'reports')
    args = parser.parse_args()
    rank_root, output, reports = args.rank_root.resolve(), args.output.resolve(), args.reports_dir.resolve()
    if output == rank_root or rank_root in output.parents and 'experiments' in output.relative_to(rank_root).parts:
        raise ValueError('Choose a new destination outside the original experiment directories')
    files, copied = build_import(rank_root, output, reports)
    for path, content in files.items():
        compatible_destination(path, content)
    for path, content in files.items():
        write_new_or_identical(path, content)
    print(json.dumps({'output': str(output), 'reports': str(reports),
                      'written_or_identical_files': len(files),
                      'copied_checkpoints': sum(item['destination'].endswith('.pt') for item in copied),
                      'output_bytes': sum(len(content) for path, content in files.items() if output in path.parents),
                      'validation_status': STATUS}, indent=2))


if __name__ == '__main__':
    main()
