#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "map_ec2_to_arc.py"


def run(matrix: str, prefix: str = "") -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT)]
    if prefix:
        cmd += ["--prefix", prefix]
    cmd.append(matrix)
    return subprocess.run(cmd, capture_output=True, text=True)


def check(condition: bool, msg: str = "") -> None:
    if not condition:
        raise AssertionError(msg)


def test_basic_matrix():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.4xlarge" },
      { config: "openreg", shard: 1, num_shards: 1, runner: "linux.2xlarge" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = json.loads(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(runners == ["l-x86iavx512-16-128", "l-x86iavx512-8-64"])


def test_matrix_with_prefix():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 7, runner: "mt-linux.4xlarge" },
      { config: "default", shard: 2, num_shards: 7, runner: "mt-linux.4xlarge" },
      { config: "openreg", shard: 1, num_shards: 1, runner: "mt-linux.2xlarge" },
    ]}"""
    result = run(matrix, prefix="mt-")
    check(result.returncode == 0, result.stderr)
    output = json.loads(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(
        runners
        == [
            "mt-l-x86iavx512-16-128",
            "mt-l-x86iavx512-16-128",
            "mt-l-x86iavx512-8-64",
        ]
    )


def test_matrix_without_prefix_when_none_present():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.g5.4xlarge.nvidia.gpu" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = json.loads(result.stdout)
    check(output["include"][0]["runner"] == "l-x86aavx2-29-113-a10g")


def test_unknown_runner_fails():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "bogus.runner" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 1)
    check("no ARC runner found for 'bogus.runner'" in result.stderr)


def test_prefix_not_present_on_runner():
    """When --prefix is given but a runner doesn't have it, the raw label is used."""
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.4xlarge" },
    ]}"""
    result = run(matrix, prefix="mt-")
    check(result.returncode == 0, result.stderr)
    output = json.loads(result.stdout)
    check(output["include"][0]["runner"] == "mt-l-x86iavx512-16-128")


def test_preserves_non_runner_fields():
    matrix = """{ include: [
      { config: "default", shard: 3, num_shards: 7, runner: "linux.large" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    entry = json.loads(result.stdout)["include"][0]
    check(entry["config"] == "default")
    check(entry["shard"] == 3)
    check(entry["num_shards"] == 7)
    check(entry["runner"] == "l-x86iavx512-2-4")


def test_empty_include_fails():
    matrix = """{ include: [] }"""
    result = run(matrix)
    check(result.returncode == 1)
    check("no 'include' entries" in result.stderr)


def test_mixed_runners():
    matrix = """{ include: [
      { config: "default", shard: 1, num_shards: 1, runner: "linux.4xlarge" },
      { config: "gpu", shard: 1, num_shards: 1, runner: "linux.g5.4xlarge.nvidia.gpu" },
      { config: "arm", shard: 1, num_shards: 1, runner: "linux.arm64.2xlarge" },
    ]}"""
    result = run(matrix)
    check(result.returncode == 0, result.stderr)
    output = json.loads(result.stdout)
    runners = [e["runner"] for e in output["include"]]
    check(
        runners
        == [
            "l-x86iavx512-16-128",
            "l-x86aavx2-29-113-a10g",
            "l-arm64g2-6-32",
        ]
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
