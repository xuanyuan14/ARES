'''
@ref: Axiomatically Regularized Pre-training for Ad hoc Search
@author: Jia Chen, Yiqun Liu, Yan Fang, Jiaxin Mao, Hui Fang, Shenghao Yang, Xiaohui Xie, Min Zhang, Shaoping Ma.
'''
from typing import Any, Iterable, List, Tuple, Union
try:
    from IPython.core.display import HTML, display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "word_attributions",
        "level",
        "rank",
        "v_q_id",
        "v_d_id",
        "doc_tokens",
        "convergence_score",
    ]

    def __init__(
        self,
        word_attributions,
        level,
        rank,
        v_q_id,
        v_d_id,
        doc_tokens,
        convergence_score,
    ):
        self.word_attributions = word_attributions
        self.level = level
        self.rank = rank
        self.v_q_id = v_q_id
        self.v_d_id = v_d_id
        self.doc_tokens = doc_tokens
        self.convergence_score = convergence_score


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    # attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 10
        sat = 75
        lig = 100-int(100 * attr)
    else:
        hue = 220
        sat = 75
        lig = 100 - int(-100 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_tooltip(item, text):
    return '<div class="tooltip">{item}\
        <span class="tooltiptext">{text}</span>\
        </div>'.format(
        item=item, text=text
    )


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        print(word, importance)
        word = format_special_tokens(word)
        color = _get_color(importance)
        # unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
        #             line-height:1.75"><font color="black"> {word}\
        #             </font></mark>'.format(
        #     color=color, word=word
        # )
        if word.startswith("##"):
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0;line-height:1.75"><font color="black">{word}</font></mark>'.format(
                color=color, word=word.replace("##","")
            )
        else:
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0;line-height:1.75"><font color="black"> {word}</font></mark>'.format(
                color=color, word=word
            )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def visualize_text(
    datarecords: Iterable[VisualizationDataRecord], legend: bool = False
    ) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = []
    dom.append("<html>")
    dom.append("<head></head>")
    dom.append("<body>")
    dom.append("<table>")
    rows = [
        '<tr><th align="left">QID DID</th>'
        '<th align="left">Relevance Level/Rank</th>'
        '<th align="left">Word Importance</th>'
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname("{}\n{}".format(datarecord.v_q_id,datarecord.v_d_id)),
                    format_classname(
                        "{}/{}".format(
                            datarecord.level, datarecord.rank
                        )
                    ),
                    format_word_importances(
                        datarecord.doc_tokens, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")
    dom.append("</body>")
    dom.append("<html>")
    html = HTML("".join(dom))
    display(html)

    return html