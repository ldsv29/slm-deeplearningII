from transformers import LlamaConfig, LlamaForCausalLM

class SLMConfig(LlamaConfig):
    model_type = "slm"

    def __init__(self, **kwargs):
        super().__init__(
            vocab_size=50304,
            hidden_size=768,
            num_hidden_layers=10,
            num_attention_heads=12,
            num_key_value_heads=4,        
            intermediate_size=2048,
            max_position_embeddings=1024,
            rope_theta=10000.0,           
            rms_norm_eps=1e-5,            
            hidden_act="silu",            
            tie_word_embeddings=True,
            **kwargs,
        )


class SLMModel(LlamaForCausalLM):
    config_class = SLMConfig

    def __init__(self, config: SLMConfig):
        super().__init__(config)
