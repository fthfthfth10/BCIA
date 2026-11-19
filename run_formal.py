"""Run the frozen G0–G2 sequence on one GPU, stopping at the first error."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bcia_peft.checkpoints import file_sha256


GROUPS = ("G0", "G1", "G2")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={Path.cwd().as_posix()}",
                                    *args], text=True).strip()


def preflight(configs, work):
    identities = {}
    commands = {}
    for group in GROUPS:
        path = configs[group]
        command = [sys.executable, "-m", "bcia_peft.train", "--config", str(path),
                   "--work-dir", str(work / group), "--device", "cuda"]
        identity = json.loads(subprocess.check_output(command + ["--describe"], text=True))
        if identity["config"]["group"] != group:
            raise ValueError("Unexpected group")
        identities[group] = identity
        commands[group] = command
    first = identities["G0"]
    common = {key: value for key, value in first["config"].items() if key != "group"}
    for identity in identities.values():
        if ({key: value for key, value in identity["config"].items() if key != "group"} != common or
                identity["initial_sha256"]["decode_head"] != first["initial_sha256"]["decode_head"] or
                any(identity[key] != first[key] for key in ("device_name", "backbone_sha256", "data_protocol"))):
            raise ValueError("Group protocols or initial heads differ")
    if identities["G1"]["initial_sha256"]["dsm_encoder"] != identities["G2"]["initial_sha256"]["dsm_encoder"]:
        raise ValueError("Initial DSM stems differ")
    return identities, commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--commit", help="Verified training source SHA")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["BCIA_DATA_ROOT"])
    if root.name != "potsdam_rgb_dsm_v3":
        raise ValueError("Formal sequence requires v3")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "verified" or metadata["version"] != root.name or any(
            file_sha256(root / "manifests" / f"{split}.jsonl") != metadata["manifest_sha256"][split]
            for split in ("train", "val")) or (
                file_sha256(root / "normalization.json") != metadata["normalization_sha256"]):
        raise ValueError("Unverified v3 data")
    if not args.describe and (not args.commit or git("rev-parse", "HEAD") != args.commit or
                              git("status", "--porcelain")):
        raise ValueError("Formal sequence requires the specified clean committed tree")
    if args.work_dir.exists():
        raise FileExistsError(args.work_dir)
    configs = {group: Path("configs") / f"formal_{group.lower()}.json" for group in GROUPS}
    identities, commands = preflight(configs, args.work_dir)
    if args.describe:
        print(json.dumps(identities, indent=2))
        return
    args.work_dir.mkdir(parents=True)
    (args.work_dir / "sequence.json").write_text(
        json.dumps(dict(commit=args.commit, groups=identities), indent=2), encoding="utf-8")
    for group in GROUPS:
        print(f"Starting {group}", flush=True)
        with (args.work_dir / f"{group}.log").open("w", encoding="utf-8") as log:
            subprocess.run(commands[group], stdout=log, stderr=subprocess.STDOUT, check=True)
        recorded = json.loads((args.work_dir / group / "run.json").read_text(encoding="utf-8"))
        if recorded != identities[group]:
            raise ValueError(f"{group} run identity differs from preflight")
        with (args.work_dir / f"{group}_summary.json").open("w", encoding="utf-8") as output:
            subprocess.run([sys.executable, "-m", "formal_results", "--work-dir",
                            str(args.work_dir / group), "--config", str(configs[group]),
                            "--backbone", os.environ["BCIA_BACKBONE_CHECKPOINT"]],
                           stdout=output, check=True)
        print(f"Verified {group} Best and Last", flush=True)


if __name__ == "__main__":
    main()
