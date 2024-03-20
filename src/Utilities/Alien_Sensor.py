def alien_sensor(bot_position: tuple[int,int], alien_positions: list[tuple[int,int]], k: int, ship_dim: int) -> bool:
    """
    :param bot_position:
    :param alien_positions:
    :param k:
    :param ship_dim:
    :return:
    """
    left = max(0, bot_position[0] - k)
    right = min(bot_position[0] + k, ship_dim - 1)
    top = max(bot_position[1] - k, 0)
    bottom = min(bot_position[1] + k, ship_dim - 1)
    for alien_position in alien_positions:
        if left <= alien_position[0] <= right and top <= alien_position[1] <= bottom:
            return True
    return False
