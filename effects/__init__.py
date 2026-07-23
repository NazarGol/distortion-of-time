"""
Effects — siblings, not a pipeline.

Each *generator* takes a clip selection plus params and yields output frames from
the pool. They are peers: superimposition and passthrough do not depend on the
weave, and the blur suite works on any of them.

    generators : weave · superimpose · passthrough ("frames")
    post stages: feedback (hyper-imposition) · blur

No registry layer, no base class — a generator is just a module with
`generate()` and `output_len()`; a post stage is just `stage(iterable, **params)`.
"""
from effects import blur, feedback, parallax, passthrough, superimpose, weave  # noqa: F401

GENERATORS = {
    weave.NAME: weave,                 # "weave"
    superimpose.NAME: superimpose,     # "superimpose"
    passthrough.NAME: passthrough,     # "frames"
}

__all__ = ["GENERATORS", "weave", "superimpose", "passthrough",
           "feedback", "blur", "parallax"]
