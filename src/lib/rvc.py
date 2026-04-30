import asyncio
import os
import shutil
import subprocess
from itertools import chain

from tqdm import tqdm

import config
import lib.ffmpeg as ffmpeg
from util import Parallel, SubprocessException, find_files, spawn


async def _poetry_get_venv(path: str):
    process = await spawn(
        "Poetry",
        "poetry",
        "env",
        "list",
        "--full-path",
        cwd=path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await process.communicate()
    return stdout.splitlines()[0].decode()


def _resolve_venv_python(venv: str):
    candidates = (
        os.path.join(venv, "Scripts", "python.exe"),
        os.path.join(venv, "Scripts", "python"),
        os.path.join(venv, "bin", "python"),
        os.path.join(venv, "python"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


async def _get_rvc_executable():
    rvc_path = os.getenv("RVC_PATH")
    venv = os.getenv("RVC_VENV")
    if venv is None or venv == "":
        venv = await _poetry_get_venv(rvc_path)

    if not os.path.isabs(venv):
        venv = os.path.join(rvc_path, venv)

    return _resolve_venv_python(venv)


def _print_torch_device(python: str, cwd: str):
    """Best effort CUDA visibility check for diagnostics."""
    cmd = [
        python,
        "-c",
        (
            "import torch;"
            "print('RVC torch:', torch.__version__);"
            "print('RVC CUDA available:', torch.cuda.is_available());"
            "print('RVC CUDA device count:', torch.cuda.device_count())"
        ),
    ]
    try:
        output = subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT)
        for line in output.strip().splitlines():
            tqdm.write(line)
    except Exception:
        tqdm.write("Could not verify CUDA availability inside RVC environment.")


class UVR:
    """UVR class."""

    process = None
    pbar = None

    def __init__(self, model: str, input_path: str, output_path: str, batchsize: int):
        self.model = model
        self.input_path = input_path
        self.output_path = output_path
        self.batchsize = batchsize

    async def start(self, total: int, title="Isolating vocals"):
        """Start the UVR process."""
        cwd = os.getcwd()

        self.pbar = tqdm(total=total, desc=title)
        python = await _get_rvc_executable()
        _print_torch_device(python, os.getenv("RVC_PATH"))
        self.process = await spawn(
            "RVC's venv python",
            python,
            os.path.join(cwd, "libs/rvc_uvr.py"),
            self.model,
            os.path.join(cwd, self.input_path),
            os.path.join(cwd, self.output_path),
            str(self.batchsize),
            cwd=os.getenv("RVC_PATH"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

    async def run(self):
        """Start checking for progress."""
        while self.process.returncode is None:
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), 5)
                file = line.decode().strip()
                if file == "##done##":
                    self.pbar.update(1)
                elif file != "":
                    tqdm.write(file)
            except asyncio.TimeoutError:
                if not self.process.stdin.is_closing():
                    # Wake up the process
                    self.process.stdin.write("\n".encode())
                    await self.process.stdin.drain()

        if self.process.returncode != 0:
            raise SubprocessException(
                "UVR process exited with code: " + str(self.process.returncode)
            )

    async def submit(self, file: str):
        """Submit a file to the UVR process."""
        self.process.stdin.write((file + "\n").encode())
        await self.process.stdin.drain()


async def isolate_vocals(
    input_path: str,
    output_path: str,
    overwrite: bool = True,
    batchsize=1,
    cache_path=config.TMP_PATH,
):
    """Splits audio files to vocals and the rest. The audio has to be correct wav."""

    # Load list of files
    files = []

    if overwrite:
        files = list(find_files(input_path))
    else:
        skipped = 0

        for file in find_files(input_path):
            filename = os.path.basename(file)
            output_filename = _get_output_filename(filename, config.UVR_FIRST_MODEL)
            if os.path.exists(os.path.join(output_path, output_filename)):
                skipped += 1
            else:
                files.append(file)

        tqdm.write(f"Skipping {skipped} already done files.")

    if len(files) == 0:
        tqdm.write("No files to process.")
        return

    # Prepare
    formatted_path = os.path.join(cache_path, "formatted")
    split_path = os.path.join(cache_path, "split")

    # Prepare conversion and splitting
    uvr_split = UVR(config.UVR_FIRST_MODEL, formatted_path, split_path, batchsize)
    ffmpegs = Parallel("[Phase 1/3] Converting files", leave=True)

    async def convert_and_process(file: str):
        dirname = os.path.dirname(file)
        os.makedirs(os.path.join(formatted_path, dirname), exist_ok=True)

        output_file = file.replace(".ogg", ".wav")
        output_path = os.path.join(formatted_path, output_file)
        if overwrite or not os.path.exists(output_path):
            await ffmpeg.to_wav(os.path.join(input_path, file), output_path)

        await uvr_split.submit(output_file)

    cached = 0

    for file in files:
        if not overwrite:
            filename = os.path.basename(file)
            dirname = os.path.dirname(file)
            output_filename = filename.replace(".ogg", ".wav")
            output_filename = _get_output_filename(
                output_filename, config.UVR_FIRST_MODEL
            )
            if os.path.exists(os.path.join(split_path, dirname, output_filename)):
                cached += 1
                continue

        ffmpegs.run(convert_and_process, file)

    if cached > 0:
        tqdm.write(f"Skipping {cached} already split files")

    # Run conversion and splitting
    await uvr_split.start(ffmpegs.count_jobs(), "[Phase 2/3] Splitting vocals")

    await asyncio.gather(ffmpegs.wait(), uvr_split.run())

    # Run dereverb
    uvr_dereverb = UVR(
        config.UVR_SECOND_MODEL, formatted_path, split_path, batchsize / 4
    )
    await uvr_dereverb.start(len(files), "[Phase 3/3] Removing reverb")
    for file in files:
        await uvr_dereverb.submit(file)

    await uvr_dereverb.run()

    tqdm.write("Cleaning up...")
    shutil.rmtree(config.TMP_PATH, ignore_errors=True)


def _get_output_filename(filename: str, model: str, instrument=False):
    """Get output filename for given input filename after processing by given model."""
    if model == "onnx_dereverb_By_FoxJoy":
        suffix = "others" if instrument else "vocal"
        return f"{filename}_{suffix}.wav"

    agg = 15  # set by the RVC script
    prefix = "instrument" if instrument else "vocal"
    return f"{prefix}_{filename}_{agg}.wav"


async def batch_rvc(input_path: str, opt_path: str, overwrite: bool, **kwargs):
    """Run RVC over given folder."""

    cwd = os.getcwd()

    tqdm.write("Starting RVC...")

    _input_path = os.path.join(cwd, input_path)
    _opt_path = os.path.join(cwd, opt_path)

    os.makedirs(_opt_path, exist_ok=True)

    files_total = sum(1 for _ in find_files(input_path))
    tqdm.write(f"Queued {files_total} files for RVC conversion.")

    python = await _get_rvc_executable()
    _print_torch_device(python, os.getenv("RVC_PATH"))

    process = await spawn(
        "RVC's venv python",
        python,
        os.path.join(cwd, "libs/infer_batch_rvc.py"),
        *("--input_path", _input_path),
        *("--opt_path", _opt_path),
        "--overwrite" if overwrite else "--no-overwrite",
        *chain(*(("--" + k, str(v)) for k, v in kwargs.items() if v is not None)),
        cwd=os.getenv("RVC_PATH"),
    )
    result = await process.wait()

    if result != 0:
        raise SubprocessException(f"Revoicing files failed with exit code {result}")
