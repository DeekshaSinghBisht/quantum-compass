"""Persistent Aer simulation worker subprocess (parent side).

All Qiskit-Aer execution in Quantum Compass is funnelled through one long-lived
child process launched with :mod:`subprocess`.  This is a stability requirement,
not an optimisation:

    Qiskit-Aer's native (C++/OpenMP) backend segfaults when a simulation runs on
    a *second* Streamlit script rerun, because each rerun executes on a fresh
    thread and the native runtime cannot be safely re-initialised across threads.
    Dragging the noise slider triggers exactly such reruns.

The child is started with ``subprocess.Popen([sys.executable, "-m",
"core._aer_worker_main"])`` — deliberately **not** ``multiprocessing`` with the
``spawn`` start method, because ``spawn`` re-imports the ``__main__`` module in
the child, and under Streamlit ``__main__`` is the page script itself (which
would re-run the whole page and recursively spawn workers → hang).

The parent exchanges length-prefixed pickled messages with the child over
stdin/stdout.  The worker is started lazily, reused for the process lifetime,
guarded by a lock, auto-restarted if it dies mid-request, and terminated on
interpreter exit.
"""

from __future__ import annotations

import atexit
import os
import pickle
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_lock = threading.RLock()
_proc: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# Framing helpers (4-byte big-endian length prefix + pickle body)
# ---------------------------------------------------------------------------
def _write_msg(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(len(data).to_bytes(4, "big"))
    stream.write(data)
    stream.flush()


def _read_exact(stream, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:  # EOF: worker died or closed the pipe
            return None
        buf.extend(chunk)
    return bytes(buf)


def _read_msg(stream):
    header = _read_exact(stream, 4)
    if header is None:
        return None
    length = int.from_bytes(header, "big")
    body = _read_exact(stream, length)
    if body is None:
        return None
    return pickle.loads(body)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def _spawn() -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # Keep the child's stdout as a clean binary protocol channel; let stderr be
    # inherited so warnings are visible but never corrupt the protocol.
    return subprocess.Popen(
        [sys.executable, "-m", "core._aer_worker_main"],
        cwd=_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
    )


def _ensure_started() -> subprocess.Popen:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    _proc = _spawn()
    return _proc


def _terminate() -> None:
    global _proc
    if _proc is None:
        return
    try:
        if _proc.poll() is None:
            try:
                _write_msg(_proc.stdin, {"op": "shutdown"})
            except Exception:
                pass
            try:
                _proc.wait(timeout=5)
            except Exception:
                _proc.kill()
    except Exception:
        pass
    finally:
        _proc = None


atexit.register(_terminate)


def _submit(op: str, payload: Dict, progress_cb: Optional[Callable[[int, int], None]] = None,
            _retried: bool = False) -> Dict:
    """Send one job to the worker and block for its single result."""
    with _lock:
        proc = _ensure_started()
        try:
            _write_msg(proc.stdin, {"op": op, "payload": payload})
        except (BrokenPipeError, OSError):
            proc = _restart_and_resend(op, payload, _retried)
            return _drain(proc, progress_cb, op, payload, _retried)
        return _drain(proc, progress_cb, op, payload, _retried)


def _restart_and_resend(op, payload, _retried) -> subprocess.Popen:
    if _retried:
        raise RuntimeError("Aer worker died repeatedly; aborting.")
    _terminate()
    proc = _ensure_started()
    _write_msg(proc.stdin, {"op": op, "payload": payload})
    return proc


def _drain(proc, progress_cb, op, payload, _retried) -> Dict:
    while True:
        msg = _read_msg(proc.stdout)
        if msg is None:  # EOF → worker died mid-request; restart once and retry
            if _retried:
                raise RuntimeError("Aer worker died mid-request; aborting.")
            _terminate()
            return _submit(op, payload, progress_cb=progress_cb, _retried=True)
        if "progress" in msg:
            if progress_cb is not None:
                progress_cb(*msg["progress"])
            continue
        if "error" in msg:
            raise RuntimeError(f"Aer worker error:\n{msg['error']}")
        return msg["result"]


# ---------------------------------------------------------------------------
# Noise descriptors
# ---------------------------------------------------------------------------
def serialize_noise(noise_model):
    """Reduce a noise descriptor to a picklable, construction-free primitive.

    Accepts:
      * ``None`` – ideal device,
      * an environment **spec dict** ``{'environment': str, 'params': dict}``,
      * a hardware **spec dict** ``{'hardware_backend': str, 'subset_size': int}``,
        whose FakeProvider model is built in the child,
      * a **``{'noise_dict': ...}``** wrapper – already-serialised Aer model,
      * a real ``qiskit_aer`` ``NoiseModel`` – passed through for CLI/tests.

    Constructing a ``NoiseModel`` in the Streamlit parent process segfaults on
    the second script rerun, so the parent must only ever pass ``None`` or a
    spec dict.
    """
    if noise_model is None:
        return None
    if isinstance(noise_model, dict):
        return noise_model
    # A real qiskit_aer NoiseModel is picklable, so pass it straight through —
    # the transport pickles it and the worker unpickles it (no deprecated
    # to_dict/from_dict round-trip).  This branch is only reached outside
    # Streamlit (CLI/tests); the page always passes None or a spec dict, so the
    # parent never constructs or serialises a qiskit object during a rerun.
    return noise_model


# ---------------------------------------------------------------------------
# Public, parent-side API (picklable primitives in / out)
# ---------------------------------------------------------------------------
def exact(n: int, edges: List, gammas: List[float], betas: List[float],
          noise_model, shots: int, readout: Optional[float]) -> Dict:
    return _submit("exact", {
        "n": n, "edges": list(edges), "gammas": [float(g) for g in gammas],
        "betas": [float(b) for b in betas], "noise_desc": serialize_noise(noise_model),
        "shots": int(shots), "readout": (None if readout is None else float(readout)),
    })


def sampling(n: int, edges: List, gammas: List[float], betas: List[float],
             noise_model, shots: int) -> Dict:
    return _submit("sampling", {
        "n": n, "edges": list(edges), "gammas": [float(g) for g in gammas],
        "betas": [float(b) for b in betas], "noise_desc": serialize_noise(noise_model),
        "shots": int(shots),
    })


def landscape(n: int, edges: List, gammas, betas, depth: int,
              noise_model, shots: int, readout: Optional[float],
              progress_cb: Optional[Callable[[int, int], None]] = None) -> Dict:
    import numpy as np

    return _submit("landscape", {
        "n": n, "edges": list(edges),
        "gammas": np.asarray(gammas, dtype=float),
        "betas": np.asarray(betas, dtype=float),
        "depth": int(depth), "noise_desc": serialize_noise(noise_model),
        "shots": int(shots), "readout": (None if readout is None else float(readout)),
    }, progress_cb=progress_cb)


def hardware_stats(backend_name: str, subset_size: Optional[int] = None) -> Dict:
    """Return IBM FakeProvider calibration stats from the isolated worker.

    The parent deliberately sends only primitive strings/integers.  This keeps
    fake-backend and Qiskit construction off Streamlit's rerun thread while the
    Hardware Profiles page can still show real calibration metadata.
    """
    return _submit("hardware_stats", {
        "backend_name": str(backend_name),
        "subset_size": (None if subset_size is None else int(subset_size)),
    })
