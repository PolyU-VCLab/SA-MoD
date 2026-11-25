from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL as Qwen3_VL_Chat
from lmms_eval.models.simple.qwen3_5_samod import Qwen3_5_SAMoD as Qwen3_5_SAMoD_Simple


@register_model("qwen3_5_samod_chat")
class Qwen3_5_SAMoD(Qwen3_VL_Chat, Qwen3_5_SAMoD_Simple):
    """Qwen3.5 SAMoD chat model: SAMoD weights plus the Qwen3.5 video generate path."""

    pass
