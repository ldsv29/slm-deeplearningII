from transformers import LlamaConfig, LlamaForCausalLM

class SLMConfig(LlamaConfig):
    model_type = "slm"

    def __init__(self, **kwargs):
        kwargs.setdefault("vocab_size", 50304)
        kwargs.setdefault("hidden_size", 768)
        kwargs.setdefault("num_hidden_layers", 10)
        kwargs.setdefault("num_attention_heads", 12)
        kwargs.setdefault("num_key_value_heads", 4)
        kwargs.setdefault("intermediate_size", 2048)
        kwargs.setdefault("max_position_embeddings", 1024)
        kwargs.setdefault("rope_theta", 10000.0)
        kwargs.setdefault("rms_norm_eps", 1e-5)
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)


class SLMModel(LlamaForCausalLM):
    config_class = SLMConfig

    def __init__(self, config: SLMConfig):
        super().__init__(config)
