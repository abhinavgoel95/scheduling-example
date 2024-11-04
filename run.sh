RUN_NAME=${1:-"testing_shard_map"}

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export VOCAB_PATH=$VOCAB_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO="Tree,Ring"  # "auto"
export NCCL_NVLS_ENABLE="0"

AR_THRESHOLD=302212254
AG_THRESHOLD=302212254
RS_THRESHOLD=50331648

NSYS_OUTPUT_FILE=/opt/haiku/profiles/testing-shard-map-${RUN_NAME}

echo "RUNNING IN PERFORMANCE MODE"

export XLA_FLAGS="
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_triton_gemm=false
--xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_enable_triton_softmax_fusion=false
--xla_gpu_all_reduce_combine_threshold_bytes=${AR_THRESHOLD}
--xla_gpu_graph_level=0
--xla_gpu_all_gather_combine_threshold_bytes=${AG_THRESHOLD}
--xla_gpu_reduce_scatter_combine_threshold_bytes=${RS_THRESHOLD}
--xla_gpu_enable_pipelined_all_gather=true
--xla_gpu_enable_pipelined_reduce_scatter=true
--xla_gpu_enable_pipelined_all_reduce=true
--xla_gpu_enable_while_loop_double_buffering=true
--xla_gpu_enable_all_gather_combine_by_dim=false
--xla_gpu_enable_reduce_scatter_combine_by_dim=false
--xla_disable_hlo_passes=rematerialization
--xla_gpu_enable_custom_fusions=false
--xla_dump_hlo_as_text --xla_dump_hlo_as_html  --xla_dump_to=/opt/haiku/hlo/hlo-shard-map-${RUN_NAME}
--xla_dump_hlo_pass_re=.*
--xla_gpu_use_memcpy_local_p2p=true
"

NSYS_CMD="nsys profile -s none -o ${NSYS_OUTPUT_FILE}-perf --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
echo ${XLA_FLAGS}
CUDA_VISIBLE_DEVICES=0,1,2,3 ${NSYS_CMD} python shard_map_all_gather.py
