from typing import Tuple, List
from util.atoms import symbol


def formatMolecule(mol: List[Tuple[int, float, float, float]]) -> str:
    output = ""
    txt = "{0}\t({1},\t{2},\t{3})\n"
    for i in range(len(mol)):
        output += txt.format(symbol[mol[i][0]], mol[i]
                             [1], mol[i][2], mol[i][3])
    return output
