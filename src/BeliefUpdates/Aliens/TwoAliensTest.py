import numpy as np
from TwoAliens import (
    initialize_belief_matrix_for_two_aliens as original_initialize,
    get_observation_matrix_for_two_aliens as original_observation,
    get_transition_prob as original_transition,
    update_belief_matrix_for_two_aliens as original_update,
    update_belief_matrix_for_two_aliens_vectorized as vectorized_update
)
from src.Utilities.Ship import Ship
from src.Utilities.utility import get_open_neighbors, open_neighbor_cells_matrix


# def test_initialize(ship_layout, bot_position, k):
#     original_belief = original_initialize(ship_layout, bot_position, k)
#     vectorized_belief = vectorized_initialize(ship_layout, bot_position, k)
#     assert np.allclose(original_belief, vectorized_belief), "Initialization failed: Matrices are not equal"
#
#
# def test_observation(ship_layout, bot_position, k, alien_sensed):
#     original_observation_matrix = original_observation(ship_layout, bot_position, k, alien_sensed)
#     vectorized_observation_matrix = vectorized_observation(ship_layout, bot_position, k, alien_sensed)
#     assert np.allclose(original_observation_matrix,
#                        vectorized_observation_matrix), "Observation failed: Matrices are not equal"


# def test_transition(open_neighbor_cells, ship_dim):
#     original_transition_matrix = original_transition(open_neighbor_cells, ship_dim)
#     vectorized_transition_matrix = vectorized_transition(open_neighbor_cells, ship_dim)
#     print(f"Transition prob check: {np.array_equal(original_transition_matrix, vectorized_transition_matrix)}")
#     assert np.allclose(original_transition_matrix,
#                        vectorized_transition_matrix), "Transition failed: Matrices are not equal"


def test_update(belief_matrix, ship_layout, bot_position, k, alien_sensed, transition_prob, open_neighbor_cells):
    original_updated_belief = original_update(belief_matrix, ship_layout, bot_position, k, alien_sensed,
                                              transition_prob, open_neighbor_cells)
    vectorized_updated_belief = vectorized_update(belief_matrix, ship_layout, bot_position, k, alien_sensed,
                                                  transition_prob, open_neighbor_cells)
    difference_matrix = original_updated_belief - vectorized_updated_belief
    discrepancy_indices = np.nonzero(difference_matrix)

    if discrepancy_indices[0].size > 0:
        print("Differences found at indices:")
        for idx in zip(*discrepancy_indices):
            print(
                f"Index: {idx}, Original: {original_updated_belief[idx]}, Vectorized: {vectorized_updated_belief[idx]}")
    else:
        print("No differences found between the original and vectorized matrices.")

    assert np.allclose(original_updated_belief, vectorized_updated_belief), "Update failed: Matrices are not equal"


def main():
    ship = Ship(35)
    ship_layout, _ = ship.generate_ship_layout()
    bot_position = (4,4)
    for row in ship_layout:
        print(row)
    k = 1
    alien_sensed = True

    # test_initialize(ship_layout, bot_position, k)
    # test_observation(ship_layout, bot_position, k, alien_sensed)

    open_neighbor_cells = open_neighbor_cells_matrix(ship_layout)
    print(open_neighbor_cells)
    ship_dim = len(ship_layout)
    # test_transition(open_neighbor_cells, ship_dim)
    ship_layout_array = np.array(ship_layout)
    belief_matrix = original_initialize(ship_layout_array, bot_position, k)
    transition_prob = original_transition(open_neighbor_cells)
    test_update(belief_matrix, ship_layout_array, bot_position, k, alien_sensed, transition_prob, open_neighbor_cells)

    print("All tests passed!")


if __name__ == "__main__":
    main()
