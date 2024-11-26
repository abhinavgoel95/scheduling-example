from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental.pjit import pjit

from jax.sharding import NamedSharding

# from ... import set_xla_metadata

num_devices = 8
devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(devices, ('i',))
# Circular permute
perm = [(i, (i + 1) % num_devices) for i in range(num_devices)]
dtype=jnp.bfloat16

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)

a = jax.device_put(jax.random.uniform(key2, (1024*8, 1024*6), dtype=dtype), NamedSharding(mesh, P('i', None)))
b = jax.device_put(jax.random.uniform(key2, (1024*6, 1024*16), dtype=dtype), NamedSharding(mesh, P(None, 'i')))

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=(P('i', None), P(None, 'i')),
         out_specs=P(None, 'i'), check_rep=False)
def collective_matmul_all_gather(a, b):
    c = jnp.zeros((num_devices, a.shape[0], b.shape[1]))
    idx = jax.lax.axis_index('i')
    for i in range(num_devices):
        with set_xla_metadata(scheduling_group_id=i):
            c_part = a @ b
            c = c.at[(idx + i) % num_devices].set(c_part)
            if i != num_devices - 1:
                a = jax.lax.ppermute(a, 'i', perm)
    return c.reshape((-1, c.shape[-1]))

for i in range(100):
    import ctypes
    libcudart = ctypes.cdll.LoadLibrary('libcudart.so')

    if i == 9:
        libcudart.cudaProfilerStart()
    if i == 12:
        libcudart.cudaProfilerStop()

    c = collective_matmul_all_gather(a, b)
    print(c.shape)
