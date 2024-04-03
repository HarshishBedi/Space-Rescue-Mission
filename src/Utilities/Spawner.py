import random
from collections import deque


class Spawner(object):
    def __init__(self, ship_layout, root_open_square):
        self.ship_layout = ship_layout
        self.root_open_square = root_open_square
        self.open_squares = self.get_open_squares()

    ''' method get_open_squares is used to fetch the list of open squares,
     which can be later used to spawn different items/characters '''

    def get_open_squares(self) -> list[tuple[int, int]]:
        # performing BFS with root_open_square as root to fetch the list of open squares
        # fringe to store all squares at a particular depth in BFS
        fringe: deque[tuple[int, int]] = deque([self.root_open_square])
        # visited_open_squares to store the visited squares during BFS
        visited_open_squares: set[tuple[int, int]] = set()
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # left, right, up, down
        # BFS
        while fringe:
            current_node = fringe.popleft()
            if current_node in visited_open_squares:
                continue
            visited_open_squares.add(current_node)
            x, y = current_node
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(self.ship_layout) and 0 <= ny < len(self.ship_layout[0]):
                    if self.ship_layout[nx][ny] == 'O' and (nx, ny) not in visited_open_squares:
                        fringe.append((nx, ny))
        # visited_open_squares should contain the entire list of open squares
        return list(visited_open_squares)

    # function for spawning bot
    def spawn_bot(self) -> tuple[list[list[int]], tuple[int, int]]:
        random_open_square_for_bot = random.choice(self.open_squares)
        x, y = random_open_square_for_bot
        self.ship_layout[x][y] = 'B'
        self.open_squares.remove(random_open_square_for_bot)
        return self.ship_layout, random_open_square_for_bot

    # function for spawning aliens
    def spawn_aliens(self, number_of_aliens: int, bot_position: tuple[int,int],
                     k:int) -> tuple[list[list[str]], list[tuple[int, int]]]:
        open_squares_available_for_aliens = self.get_open_squares_for_aliens(bot_position, k)
        random_open_squares_for_aliens = random.sample(open_squares_available_for_aliens, number_of_aliens)
        for alien in random_open_squares_for_aliens:
            self.ship_layout[alien[0]][alien[1]] = 'A'
        return self.ship_layout, random_open_squares_for_aliens

    # function for spawning crew members
    def spawn_crew_members(self, number_of_crew_members) -> tuple[list[list[str]], list[tuple[int, int]]]:
        random_open_squares_for_crew_members = random.sample(self.open_squares, number_of_crew_members)
        for crew in random_open_squares_for_crew_members:
            x, y = crew
            if self.ship_layout[x][y] == 'A':
                self.ship_layout[x][y] = 'CM&A'
            elif self.ship_layout[x][y] == 'A&A':
                self.ship_layout[x][y] = 'CM&A&A'
            elif self.ship_layout[x][y] == 'O':
                self.ship_layout[x][y] = 'CM'
            else:
                print('Some issue in the ship layout while spawning the crew member')
                self.ship_layout[x][y] = 'CM'
        return self.ship_layout, random_open_squares_for_crew_members

    def get_open_squares_for_aliens(self, bot_position: tuple[int,int], k:int)-> list[tuple[int,int]]:
        open_squares_for_aliens = self.open_squares.copy()
        left = max(0, bot_position[0] - k)
        right = min(bot_position[0] + k, len(self.ship_layout) - 1)
        top = max(bot_position[1] - k, 0)
        bottom = min(bot_position[1] + k, len(self.ship_layout) - 1)
        for i in range(left, right + 1):
            for j in range(top, bottom + 1):
                if (i, j) in open_squares_for_aliens:
                    open_squares_for_aliens.remove((i, j))
        return open_squares_for_aliens
