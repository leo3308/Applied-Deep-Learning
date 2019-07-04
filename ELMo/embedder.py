import numpy as np
import json
import pickle
from ELMo.ELMoForManyLangs.elmoformanylangs import embedder
from ELMo.sent2elmo import sent2elmo


class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        self.model_path = 'ELMo/final/model_0'
        
        config_path = 'ELMo/config/config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        char_lexicon_path = 'ELMo/final/charLexicon.pkl'
        with open(char_lexicon_path, 'rb') as f:
            self.char_lexicon = pickle.load(f)

        self.device = 'cuda:0'

        self.elmo = sent2elmo(self.char_lexicon, self.config, self.device, self.model_path)

        # TODO

    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        # TODO
        max_len = min(max(map(len, sentences)), max_sent_len)
        sentence = []
        for sent in sentences:
            while True:
                if len(sent) > max_len:
                    sent.pop()
                elif len(sent) < max_len:
                    sent.append('<pad>')
                else:
                    break
            sent.append('<eos>')
            sent.insert(0, '<bos>')
            sentence.append(sent)
    
        features = self.elmo.get_feature(sentence)
        features = features.detach().cpu().numpy()
        features = np.expand_dims(features, axis=2)

        return features

