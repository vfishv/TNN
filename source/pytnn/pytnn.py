from _pytnn import *
from typing import List, Dict, Any

def _supported_input_size_type(input_size) -> bool:
    if isinstance(input_size, tuple):
        return True
    elif isinstance(input_size, list):
        return True
    else:
        raise TypeError(
            "Input sizes for inputs are required to be a List, tuple or a Dict of two sizes (min, max), found type: "
            + str(type(input_size)))


def _parse_input_ranges(input_sizes: List):
    if any(not isinstance(i, dict) and not _supported_input_size_type(i) for i in input_sizes):
        raise KeyError("An input size must either be a static size or a range of two sizes (min, max) as Dict")
    min_input_shapes = {}
    max_input_shapes = {}
    for index, value in enumerate(input_sizes):
        if isinstance(value, dict):
            if all(k in value for k in ["min", "max"]):
                min_input_shapes["input_" + str(index)] = value["min"]
                max_input_shapes["input_" + str(index)] = value["max"]
            else:
                raise KeyError(
                    "An input size must either be a static size or a range of three sizes (min, opt, max) as Dict")
        elif isinstance(value, list):
            min_input_shapes["input_" + str(index)] = value
            max_input_shapes["input_" + str(index)] = value
        elif isinstance(value, tuple):
            min_input_shapes["input_" + str(index)] = value
            max_input_shapes["input_" + str(index)] = value
    return (min_input_shapes, max_input_shapes)


def _parse_device_type(device_type):
    if isinstance(device_type, DeviceType):
        return device_type
    elif isinstance(device_type, str):
        if device_type == "gpu" or device_type == "GPU" or device_type == "CUDA" or device_type == "cuda":
            return DEVICE_CUDA
        elif device_type == "cpu" or device_type == "CPU" or device_type == "X86" or device_type == "x86":
            return DEVICE_X86
        elif device_type == "arm" or device_type == "ARM":
            return DEVICE_ARM
        elif device_type == "naive" or device_type == "NAIVE":
            return DEVICE_NAIVE
        elif device_type == "metal" or device_type == "METAL":
            return DEVICE_METAL
        elif device_type == "opencl" or device_type == "OPENCL":
            return DEVICE_OPENCL
        else:
            ValueError("Got a device_type unsupported (type: " + device_type + ")")
    else:
        raise TypeError("device_type must be of type string or DeviceType, but got: " +
                        str(type(device_type)))         

def _parse_network_type(network_type):
    if isinstance(network_type, NetworkType):
        return network_type
    elif isinstance(network_type, str):
        if network_type == "auto" or network_type == "AUTO":
            return NETWORK_TYPE_AUTO
        elif network_type == "default" or network_type == "DEFAULT":
            return NETWORK_TYPE_DEFAULT
        elif network_type == "openvino" or network_type == "OPENVINO":
            return NETWORK_TYPE_OPENVINO
        elif network_type == "coreml" or network_type == "COREML":
            return NETWORK_TYPE_COREML
        elif network_type == "tensorrt" or network_type == "TENSORRT":
            return NETWORK_TYPE_TENSORRT
        elif network_type == "tnntorch" or network_type == "ATLAS":
            return NETWORK_TYPE_TNNTORCH
        elif network_type == "atlas" or network_type == "ATLAS":
            return NETWORK_TYPE_ATLAS
        else:
            ValueError("Got a network_type unsupported (type: " + network_type + ")")
    else:
        raise TypeError("network_type must be of type string or NetworkType, but got: " +
                        str(type(network_type)))

def _parse_precision(precision):
    if isinstance(precision, Precision):
        return precision
    elif isinstance(precision, str):
        if precision == "auto" or precision == "AUTO":
            return PRECISION_AUTO
        if precision == "normal" or precision == "NORMAL":
            return PRECISION_NORMAL
        elif precision == "high" or precision == "HIGH" or precision == "fp32" or precision == "FP32" \
            or precision == "float32" or precision == "FLOAT32":
            return PRECISION_HIGH
        elif precision == "low" or precision == "LOW" or precision == "fp16" or precision == "FP16" \
            or precision == "float16" or precision == "FLOAT16" or precision == "bfp16" or precision == "BFP16":
            return PRECISION_LOW

def _parse_network_config(config_dict):
    network_config = NetworkConfig()
    if "device_type" in config_dict:
        network_config.device_type = _parse_device_type(config_dict["device_type"])
    else:
        network_config.device_type = DEVICE_CUDA
    if "device_id" in config_dict:
        assert isinstance(config_dict["device_id"], int)
        network_config.device_id = config_dict["device_id"]
    if "data_format" in config_dict:
        assert isinstance(config_dict["data_format"], DataFormat)
        network_config.data_format = config_dict["data_format"]
    if "network_type" in config_dict:
        network_config.network_type = _parse_network_type(config_dict["network_type"])
    if "share_memory_mode" in config_dict:
        assert isinstance(config_dict["share_memory_mode"], ShareMemoryMode)
        network_config.share_memory_mode = config_dict["share_memory_mode"]
    if "library_path" in config_dict:
        network_config.library_path = config_dict["library_path"]
    if "precision" in config_dict:
        network_config.precision = _parse_precision(config_dict["precision"])
    if "cache_path" in config_dict:
        network_config.cache_path = config_dict["cache_path"]
    if "enable_tune_kernel" in config_dict:
        network_config.enable_tune_kernel = config_dict["enable_tune_kernel"]
    return network_config

def _replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

def load(model_path, config_dict = {}):
    min_input_shapes = None
    max_input_shapes = None
    if "input_shapes" in config_dict:
        min_input_shapes, max_input_shapes = _parse_input_ranges(config_dict["input_shapes"])
    network_config = _parse_network_config(config_dict)
    return Module(model_path, network_config, min_input_shapes, max_input_shapes)

def load_raw(model_path, network_config=None, input_shapes=None):
    return Module(model_path, network_config, input_shapes, input_shapes)

def load_raw_range(model_path, network_config=None, min_input_shapes=None, max_input_shapes=None):
    return Module(model_path, network_config, min_input_shapes, max_input_shapes)

class Module:
    def __init__(self, model_path, network_config, min_input_shapes, max_input_shapes):
        self.model_path = model_path
        self.tnn=TNN()
        model_config=ModelConfig()
        if model_path.endswith("tnnproto"):
            weights_path=_replace_last(model_path, "tnnproto", "tnnmodel")
            model_config.model_type=MODEL_TYPE_TNN
            params = []
            with open(model_path, "r") as f:
                params.append(f.read())
            with open(weights_path, "rb") as f:
                params.append(f.read())
            model_config.params=params
        else:
            model_config.model_type=MODEL_TYPE_TORCHSCRIPT
            model_config.params=[model_path]
        self.tnn.Init(model_config)
        ret=Status()
        if network_config is None:
            network_config=NetworkConfig()
            network_config.device_type=DEVICE_CUDA
        if model_config.model_type == MODEL_TYPE_TORCHSCRIPT:
            network_config.network_type=NETWORK_TYPE_TNNTORCH
        if min_input_shapes is None:
            self.instance=self.tnn.CreateInst(network_config, ret)
        elif max_input_shapes is None:
            self.instance=self.tnn.CreateInst(network_config, ret, min_input_shapes)
        else:
            self.instance=self.tnn.CreateInst(network_config, ret, min_input_shapes, max_input_shapes)

    def forward(self, *inputs, rtype="list"):
        if len(inputs) > 1:
            for index, value in enumerate(inputs):
                self.instance.SetInputMat(convert_numpy_to_mat(value), MatConvertParam(), "input_" + str(index))
        else:
            if isinstance(inputs[0], tuple) or isinstance(inputs[0], list):
                for index, value in enumerate(inputs[0]):
                    self.instance.SetInputMat(convert_numpy_to_mat(value), MatConvertParam(), "input_" + str(index))
            elif isinstance(inputs[0], dict):
                for key, value in inputs[0].items():
                    self.instance.SetInputMat(convert_numpy_to_mat(value), MatConvertParam(), key)
            else:
                self.instance.SetInputMat(convert_numpy_to_mat(inputs[0]), MatConvertParam()) 
        self.instance.Forward()
        output_blobs = self.instance.GetAllOutputBlobs()
        output = []
        is_dict = False
        if rtype == "dict":
            output = {}
            is_dict = True
        for key, value in output_blobs.items():
            output_mat=self.instance.GetOutputMat(MatConvertParam(), key, DEVICE_NAIVE, NCHW_FLOAT)
            output_mat_numpy=convert_mat_to_numpy(output_mat)
            if is_dict:
                output[key] = output_mat_numpy
            else:
                output.append(output_mat_numpy)
        return output