import cv2


def test_card(icon_file_path, suit_file_path):
    card_icon_templates = {
        '♣': cv2.imread('../images/Clover.png', cv2.IMREAD_GRAYSCALE),  # Clubs
        '♦': cv2.imread('../images/Diamonds.png', cv2.IMREAD_GRAYSCALE),  # Diamonds
        '♥': cv2.imread('../images/Hearts.png', cv2.IMREAD_GRAYSCALE),  # Hearts
        '♠': cv2.imread('../images/Spades.png', cv2.IMREAD_GRAYSCALE)  # Spades
    }

    card_number_templates = {
        '2': cv2.imread('../images/2.png', cv2.IMREAD_GRAYSCALE),
        '3': cv2.imread('../images/3.png', cv2.IMREAD_GRAYSCALE),
        '4': cv2.imread('../images/4.png', cv2.IMREAD_GRAYSCALE),
        '5': cv2.imread('../images/5.png', cv2.IMREAD_GRAYSCALE),
        '6': cv2.imread('../images/6.png', cv2.IMREAD_GRAYSCALE),
        '7': cv2.imread('../images/7.png', cv2.IMREAD_GRAYSCALE),
        '8': cv2.imread('../images/8.png', cv2.IMREAD_GRAYSCALE),
        '9': cv2.imread('../images/9.png', cv2.IMREAD_GRAYSCALE),
        '10': cv2.imread('../images/10.png', cv2.IMREAD_GRAYSCALE),
        'A': cv2.imread('../images/A.png', cv2.IMREAD_GRAYSCALE),
        'J': cv2.imread('../images/J.png', cv2.IMREAD_GRAYSCALE),
        'Q': cv2.imread('../images/Q.png', cv2.IMREAD_GRAYSCALE),
        'K': cv2.imread('../images/K.png', cv2.IMREAD_GRAYSCALE)
    }

    loaded_icon_gray = cv2.imread(suit_file_path, cv2.IMREAD_GRAYSCALE)
    loaded_num_gray = cv2.imread(icon_file_path, cv2.IMREAD_GRAYSCALE)

    best_card_suit = None
    best_card_suit_value = 0
    for suit, template in card_icon_templates.items():
        result = cv2.matchTemplate(loaded_icon_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_card_suit_value:
            best_card_suit = suit
            best_card_suit_value = max_val

    best_card_rank = None
    best_card_rank_value = 0
    for rank, template in card_number_templates.items():
        result = cv2.matchTemplate(loaded_num_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_card_rank_value:
            best_card_rank = rank
            best_card_rank_value = max_val

    if best_card_suit:
        print('Found suit %s' % best_card_suit)
    if best_card_rank:
        print('Found rank %s' % best_card_rank)


test_card('../tmp/king.png', '../tmp/diamond.png')
