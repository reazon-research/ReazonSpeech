from transformers import PretrainedConfig


class ResEncoderConfig(PretrainedConfig):
    model_type = "modified_resnet"

    def __init__(
        self,
        relu_type="prelu",
        frontend_nout=64,
        backend_out=512,
        **kwargs,
    ):
        self.relu_type = relu_type
        self.frontend_nout = frontend_nout
        self.backend_out = backend_out
        super().__init__(**kwargs)
