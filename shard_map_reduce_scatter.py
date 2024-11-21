from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental.pjit import pjit

from jax.sharding import NamedSharding

num_devices = 4
devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(devices, ('i',))

dtype=jnp.bfloat16

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)

a = jax.device_put(jax.random.uniform(key2, (1024*8, 1024*16), dtype=dtype), NamedSharding(mesh, P(None, 'i')))
b = jax.device_put(jax.random.uniform(key2, (1024*16, 1024*6), dtype=dtype), NamedSharding(mesh, P('i', None)))

# Circular permute
perms = [
    [(i, (i - j) % num_devices) for i in range(num_devices)] for j in range(1, num_devices)
]

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=(P(None, 'i'), P('i', None)),
         out_specs=P(None, None), check_rep=False)
def collective_matmul_reduce_scatter(a, b):
    c = jnp.zeros((a.shape[0]//num_devices, b.shape[1]), dtype=dtype)
    c1 = jnp.zeros((a.shape[0]//num_devices, b.shape[1]), dtype=dtype)
    out = jnp.zeros((a.shape[0]//num_devices, b.shape[1]), dtype=dtype)
    out1 = jnp.zeros((a.shape[0]//num_devices, b.shape[1]), dtype=dtype)

    idx = jax.lax.axis_index('i')
    a = a.reshape(num_devices, a.shape[0]//num_devices, -1)

    for i in range(num_devices//2):
        slice_a = a[(idx + (num_devices - 1) - 2*i) % num_devices]
        out = jnp.dot(slice_a, b)
        c += jax.lax.ppermute(out, 'i', perms[2*i])

        slice_a1 = a[(idx + (num_devices - 1) - (2*i+1)) % num_devices]
        out1 = jnp.dot(slice_a1, b)

        if (2*i+1) != num_devices - 1:
            c1 += jax.lax.ppermute(out1, 'i', perms[2*i+1])
    return c + c1 + out1

for i in range(100):
    import ctypes
    libcudart = ctypes.cdll.LoadLibrary('libcudart.so')

    if i == 9:
        libcudart.cudaProfilerStart()
    if i == 12:
        libcudart.cudaProfilerStop()

    d = collective_matmul_reduce_scatter(a, b)
    print(d)
