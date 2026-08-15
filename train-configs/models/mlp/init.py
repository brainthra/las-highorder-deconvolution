# check if variable configs exists else set a blank dict
try:
    configs
except NameError:
    configs = {}

layer_nodes = configs.get("layer_nodes", [32, 64, 128, 256])

kwargs = dict(
    input_size=design.outputs,
    output_size=len(sample_idxs),
    layer_nodes=layer_nodes
)

model = get_model("mlp", **kwargs)