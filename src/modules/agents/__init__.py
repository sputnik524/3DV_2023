REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rlsac_agent import SACAgent, SACAgent_sharedbase
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["sac_ns"] = SACAgent
REGISTRY["marlsac_sharedbase"] = SACAgent_sharedbase