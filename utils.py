def build_wcst_token_map():
    """Builds a dictionary mapping WCST token IDs (0–69) to readable string labels."""
    colours = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'star', 'cross']
    quantities = ['1', '2', '3', '4']
    categories = ['C1', 'C2', 'C3', 'C4']

    token_map = {}

    # 0 = PAD
    token_map[0] = "PAD"

    # 1–64: all possible card combos
    idx = 1
    for colour in colours:
        for shape in shapes:
            for quantity in quantities:
                token_map[idx] = f"{colour}-{shape}-{quantity}"
                idx += 1

    # 64–67: categories
    for i, c in enumerate(categories, start=64):
        token_map[i] = c

    # 68, 69: special tokens
    token_map[68] = "SEP"
    token_map[69] = "EOS"

    return token_map
