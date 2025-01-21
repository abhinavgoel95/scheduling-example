# Collective Matmul in Shardmap Experimental Implementations

These scripts are experimental and intended for testing new features to enable building efficient collective matmuls from shard_map.

To get started, first you should ensure you are using the correct XLA and JAX patches while this feature remains experimental

### XLA

https://github.com/chaserileyroberts/xla
Branch: chase/xai_stream_annotation_patch

### JAX

https://github.com/chaserileyroberts/jax
Branch: chase/xai_stream_annotation_patch

## Testing scripts

We recommend using `nsys-jax` for profiling.

```bash
nsys-jax python annotated_collective_matmul.py
```

## How to use scheduling annotation

Scheduling annotations can be applied to operations with the `set_xla_metadata(_scheduling_group_id=...)` context.

Example:

```python
from jax._src.xla_metadata import set_xla_metadata


with set_xla_metadata(_scheduling_group_id=...):
   a = jax.lax.ppermute(a, ...)
```

Operations that are under the same scheduling group will be launched in parallel when possible. All async operations within 
the same scheduling group will also be `done`d as a group aswell.

## How to use stream annotation

Stream annotations can be applied by using the `compute_on` transform. The API is

```python
streamed_fn = compute_on(f"stream:{STREAM_ID}:{SCHEDULING_GROUP}")(jit_subroutine)
```

Where `STREAM_ID` is the stream you want to launch on, and `SCHEDULING_GROUP` is the scheduling group you want to launch with.
Stream annotation can only be applied to subroutines that are also decorated with a `jax.jit`

Example:

```python
from jax.experimental.compute_on import compute_on

@jax.jit
def gemm(a, b):
    return a @ b

c = compute_on("stream:1:3")(gemm)(a=..., b=...)
```

In the above example, the gemm will run on stream #1 and be launched with scheduling group #3.

* NOTE: You can not (yet) apply scheduling annotation via `set_xla_metadata` with the stream annotation. Instead, you must use the scheduling group id from the `compute_on` definition.


## Known issues

* Stream annotation only works for gemms. Including other operations like an `add` or `dynamic_update_slice` will cause assertion errors during compilation. This is our next main focus area to get working.
* Interweaved scheduling (i.e., scheduling like `start1, start2, done1, start3, done2, start4, done3, ...`) isn't directly possible with the current scheduling system, as all grouped `starts` must be `done`ed before the next group will launch.
