import re

import nltk





def detect_noun_phrases(sentence: str) -> list[str]:
    """
    Extract noun phrases from a sentence

    Returns a list of nouns phrases extracted from a given sentence

    Parameters
    ----------
    sentence : str
        The argument of a function

    Returns
    -------
    list[str]
    
    Examples
    --------
    >>> print(detect_noun_phrase("I am a project manager"))
    ['project manager']
    """
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    chunk = r"""Chink:
                        {<NN><WP|IN|VB.?>}
                Chunk:
                        {<JJ>}
                        {<NN><NN>}
                        {<NN.?>*}
                        {<VB.?><DT>?<NN>}"""
    chunkParser = nltk.RegexpParser(chunk)
    chunked = chunkParser.parse_all(tags)

    result = []
    for a in chunked:
        if isinstance(a, nltk.tree.Tree):
            if a.label() == 'Chunk':
                result.append(" ".join([lf[0].lower() for lf in a.leaves()]))
    return result