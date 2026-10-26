"""Standalone entry point for the Aer simulation worker subprocess.

This is launched by :mod:`core.aer_worker` as::

    python -m core._aer_worker_main

It speaks a simple length-prefixed pickle protocol over stdin/stdout so it can
run Qiskit-Aer simulations in a process that is fully isolated from Streamlit's
rotating script-runner threads (which otherwise segfault Aer's native backend on
the second rerun).

Protocol
--------
Request  (parent → child, on stdin):  ``{'op': str, 'payload': dict}``
Response (child → parent, on stdout): zero or more ``{'progress': (done, total)}``
followed by exactly one ``{'result': dict}`` or ``{'error': str}``.

To keep stdout a clean protocol channel, any accidental library ``print`` is
redirected to stderr by reassigning ``sys.stdout`` after duplicating the real
stdout fd for protocol use.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when launched with an absolute path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _write_exact(stream, data: bytes) -> None:
    stream.write(len(data).to_bytes(4, "big"))
    stream.write(data)
    stream.flush()


def _read_exact(stream, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:  # EOF
            return None
        buf.extend(chunk)
    return bytes(buf)


def _read_msg(stream):
    import pickle

    header = _read_exact(stream, 4)
    if header is None:
        return None
    length = int.from_bytes(header, "big")
    body = _read_exact(stream, length)
    if body is None:
        return None
    return pickle.loads(body)


def main() -> int:
    import pickle

    # Dedicated, unbuffered protocol stream on the real stdout fd; send any
    # stray library prints to stderr so they never corrupt the protocol.
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    sys.stdout = sys.stderr
    stdin = sys.stdin.buffer

    from qiskit_aer.noise import NoiseModel

    from core import _qaoa_sim as sim
    from core.hardware_profiles import extract_hardware_stats, get_hardware_noise_model
    from core.noise_models import get_noise_model

    def rebuild_noise(desc):
        """Reconstruct a NoiseModel (or None) from a descriptor.

        Accepts ``None``, a spec dict ``{'environment', 'params'}``, a legacy
        ``{'noise_dict': ...}`` wrapper, or an already-pickled ``NoiseModel``
        instance (passed straight through by the parent).
        """
        if desc is None:
            return None
        if isinstance(desc, dict):
            if "noise_dict" in desc:
                return NoiseModel.from_dict(desc["noise_dict"])
            if "environment" in desc:
                return get_noise_model(desc["environment"], desc.get("params"))
            if "hardware_backend" in desc:
                # Hardware fake backends and NoiseModels are constructed only
                # in this child process, never on Streamlit's parent thread.
                return get_hardware_noise_model(
                    desc["hardware_backend"], desc.get("subset_size")
                )
            raise ValueError(f"Unrecognised noise descriptor: {desc!r}")
        # Already a NoiseModel instance (unpickled from the parent).
        return desc

    def resolve_readout(p, noise_model):
        r = p.get("readout")
        if r is not None:
            return float(r)
        return sim.readout_prob_from_model(noise_model)

    def send(obj):
        _write_exact(proto_out, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    while True:
        msg = _read_msg(stdin)
        if msg is None:  # parent closed the pipe
            break
        op = msg.get("op")
        if op == "shutdown":
            break
        try:
            p = msg["payload"]
            noise_model = rebuild_noise(p.get("noise_desc"))

            if op == "exact":
                out = sim.run_exact(
                    p["n"], p["edges"], p["gammas"], p["betas"],
                    noise_model, p["shots"], resolve_readout(p, noise_model),
                )
            elif op == "sampling":
                out = sim.run_sampling(
                    p["n"], p["edges"], p["gammas"], p["betas"],
                    noise_model, p["shots"],
                )
            elif op == "landscape":
                def _cb(done, total):
                    send({"progress": (int(done), int(total))})

                out = sim.run_landscape(
                    p["n"], p["edges"], p["gammas"], p["betas"], p["depth"],
                    noise_model, p["shots"], resolve_readout(p, noise_model),
                    progress_cb=_cb,
                )
            elif op == "hardware_stats":
                # Pure primitive result back to the parent; FakeProvider
                # construction remains isolated in this worker.
                out = extract_hardware_stats(p["backend_name"], p.get("subset_size"))
            else:
                raise ValueError(f"unknown op '{op}'")
            send({"result": out})
        except Exception:
            import traceback

            send({"error": traceback.format_exc()})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
