import util
import os, sys
import datetime, time
import argparse
import signal

class SokobanState:
    def __init__(self, player, boxes):
        self.data = tuple([player] + sorted(boxes))
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None

    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data

    def __lt__(self, other):
        return self.data < other.data

    def __hash__(self):
        return hash(self.data)

    def player(self):
        return self.data[0]

    def boxes(self):
        return self.data[1:]

    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved

    def act(self, problem, act):
        if act in self.adj:
            return self.adj[act]
        else:
            val = problem.valid_move(self, act)
            self.adj[act] = val
            return val

    def box_is_cornered(self, map, box, targets, all_boxes):

        def row_is_trap(offset):
            target_count = 0
            box_count = 1
            for direction in [-1, 1]:
                index = box[1] + direction
                while not map[box[0]][index].wall:
                    if map[box[0] + offset][index].floor:
                        return None
                    elif map[box[0]][index].target:
                        target_count += 1
                    elif (box[0], index) in all_boxes:
                        box_count += 1
                    index += direction

            if box_count > target_count:
                return True
            return None

        def column_is_trap(offset):
            target_count = 0
            box_count = 1
            for direction in [-1, 1]:
                index = box[0] + direction
                while not map[index][box[1]].wall:
                    if map[index][box[1] + offset].floor:
                        return None
                    elif map[index][box[1]].target:
                        target_count += 1
                    elif (index, box[1]) in all_boxes:
                        box_count += 1
                    index += direction

            if box_count > target_count:
                return True
            return None

        if box not in targets:
            if map[box[0] - 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] - 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True

            if map[box[0] - 1][box[1]].wall:
                if row_is_trap(offset=-1):
                    return True
            elif map[box[0] + 1][box[1]].wall:
                if row_is_trap(offset=1):
                    return True
            elif map[box[0]][box[1] - 1].wall:
                if column_is_trap(offset=-1):
                    return True
            elif map[box[0]][box[1] + 1].wall:
                if column_is_trap(offset=1):
                    return True

        return None

    def adj_box(self, box, all_boxes):
        adj = []
        for i in all_boxes:
            if box[0] - 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[0] + 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[1] - 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
            elif box[1] + 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
        return adj

    def box_is_trapped(self, map, box, targets, all_boxes):
        if self.box_is_cornered(map, box, targets, all_boxes):
            return True

        adj_boxes = self.adj_box(box, all_boxes)
        for i in adj_boxes:
            if box not in targets and i not in targets:
                if i['direction'] == 'vertical':
                    if map[box[0]][box[1] - 1].wall and map[i['box'][0]][i['box'][1] - 1].wall:
                        return True
                    elif map[box[0]][box[1] + 1].wall and map[i['box'][0]][i['box'][1] + 1].wall:
                        return True
                if i['direction'] == 'horizontal':
                    if map[box[0] - 1][box[1]].wall and map[i['box'][0] - 1][i['box'][1]].wall:
                        return True
                    elif map[box[0] + 1][box[1]].wall and map[i['box'][0] + 1][i['box'][1]].wall:
                        return True

        return None

    def deadp(self, problem):
        temp_boxes = self.data[1:]
        for box in list(temp_boxes):
            if self.box_is_trapped(problem.map, box, problem.targets, temp_boxes):
                self.dead = True
        return self.dead

    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache


class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target


def parse_move(move):
    if move == 'u':
        return (-1, 0)
    elif move == 'd':
        return (1, 0)
    elif move == 'l':
        return (0, -1)
    elif move == 'r':
        return (0, 1)
    raise Exception('Invalid move character.')


class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class SokobanProblem(util.SearchProblem):
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0, 0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)

    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map) - 1, len(self.map[-1]) - 1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row, col) in s.boxes()
                player = (row, col) == s.player()
                if box and target:
                    print(DrawObj.BOX_ON, end='')
                elif player and target:
                    print(DrawObj.PLAYER, end='')
                elif target:
                    print(DrawObj.TARGET, end='')
                elif box:
                    print(DrawObj.BOX_OFF, end='')
                elif player:
                    print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall:
                    print(DrawObj.WALL, end='')
                else:
                    print(DrawObj.FLOOR, end='')
            print()

    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        dx, dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1, y1) in s.boxes():
            if self.map[x2][y2].floor and (x2, y2) not in s.boxes():
                return True, True, SokobanState((x1, y1),
                                                [b if b != (x1, y1) else (x2, y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1, y1), s.boxes())

    def dead_end(self, s):
        if not self.dead_detection:
            return False

        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):
        if self.dead_end(s):
            return []
        return s.all_adj(self)

    def display_state(self, s):
        self.print_state(s)


class SokobanProblemFaster(SokobanProblem):
    def flood_fill(self, problem, matrix, path_list, current_path, x, y):
        box_pos = problem.data[1:]
        if matrix[x][y].floor and not matrix[x][y].visited:
            matrix[x][y].visited = True

            if (x - 1, y) in box_pos:
                if not matrix[x - 2][y].wall and (x - 2, y) not in box_pos:
                    path_list.append(current_path + 'u')
            if (x + 1, y) in box_pos:
                if not matrix[x + 2][y].wall and (x + 2, y) not in box_pos:
                    path_list.append(current_path + 'd')
            if (x, y - 1) in box_pos:
                if not matrix[x][y - 2].wall and (x, y - 2) not in box_pos:
                    path_list.append(current_path + 'l')
            if (x, y + 1) in box_pos:
                if not matrix[x][y + 2].wall and (x, y + 2) not in box_pos:
                    path_list.append(current_path + 'r')

            if not matrix[x - 1][y].wall and (x - 1, y) not in box_pos and not matrix[x - 1][y].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'u', x - 1, y)
            if not matrix[x + 1][y].wall and (x + 1, y) not in box_pos and not matrix[x + 1][y].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'd', x + 1, y)
            if not matrix[x][y - 1].wall and (x, y - 1) not in box_pos and not matrix[x][y - 1].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'l', x, y - 1)
            if not matrix[x][y + 1].wall and (x, y + 1) not in box_pos and not matrix[x][y + 1].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'r', x, y + 1)

            return path_list

        return path_list

    def get_position_from_path(self, player, path):
        for move in path:
            if move == 'u':
                player = (player[0] - 1, player[1])
            elif move == 'd':
                player = (player[0] + 1, player[1])
            elif move == 'l':
                player = (player[0], player[1] - 1)
            elif move == 'r':
                player = (player[0], player[1] + 1)
        return player

    def expand(self, s):
        if self.dead_end(s):
            return []

        for i in self.map:
            for j in i:
                j.visited = False

        path_list = self.flood_fill(s, self.map, list(), '', s.data[0][0], s.data[0][1])

        new_states = []
        for path in path_list:
            new_player = self.get_position_from_path(s.data[0], path)

            box_index = list(s.data[1:]).index(new_player)
            new_boxes = list(s.data[1:])
            if path[-1] == 'u':
                new_boxes[box_index] = (new_boxes[box_index][0] - 1, new_boxes[box_index][1])
            elif path[-1] == 'd':
                new_boxes[box_index] = (new_boxes[box_index][0] + 1, new_boxes[box_index][1])
            elif path[-1] == 'l':
                new_boxes[box_index] = (new_boxes[box_index][0], new_boxes[box_index][1] - 1)
            elif path[-1] == 'r':
                new_boxes[box_index] = (new_boxes[box_index][0], new_boxes[box_index][1] + 1)

            new_states.append(
                (path, SokobanState(player=new_player, boxes=new_boxes), len(path)))

        return new_states


class Heuristic:
    def __init__(self, problem):
        self.problem = problem
        self.buff = self.calc_cost()
        self.box_state = self.problem.init_boxes
        self.memo = dict()

    def calc_manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def heuristic(self, s):
        box_pos = s.data[1:]
        targets = self.problem.targets
        targets_left = len(targets)
        total = 0
        for ind, box in enumerate(box_pos):
            total += self.calc_manhattan(box, targets[ind])
            if box in targets:
                targets_left -= 1
        return total * targets_left

    def calc_cost(self):

        def flood(x, y, cost):
            if not visited[x][y]:

                if buff[x][y] > cost:
                    buff[x][y] = cost
                visited[x][y] = True

                if self.problem.map[x - 1][y].floor:
                    flood(x - 1, y, cost + 1)
                if self.problem.map[x + 1][y].floor:
                    flood(x + 1, y, cost + 1)
                if self.problem.map[x][y - 1].floor:
                    flood(x, y - 1, cost + 1)
                if self.problem.map[x][y + 1].floor:
                    flood(x, y + 1, cost + 1)

        buff = [[float('inf') for _ in j] for j in self.problem.map]
        for target in self.problem.targets:
            visited = [[False for _ in i] for i in self.problem.map]
            flood(target[0], target[1], 0)

        return buff

    def box_moved(self, current):
        count = 0
        for ind, val in enumerate(current):
            if val != self.box_state[ind]:
                count += 1
        self.box_state = current
        return count

    def heuristic2(self, s):
        box_pos = s.data[1:]
        if box_pos in self.memo:
            return self.memo[box_pos]
        targets = self.problem.targets
        matrix = self.problem.map
        box_moves = self.box_moved(box_pos)
        total = 0
        targets_left = len(targets)
        for val in box_pos:
            if val not in targets:
                if matrix[val[0] - 1][val[1]].wall and matrix[val[0]][val[1] - 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
                elif matrix[val[0] - 1][val[1]].wall and matrix[val[0]][val[1] + 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
                elif matrix[val[0] + 1][val[1]].wall and matrix[val[0]][val[1] - 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
                elif matrix[val[0] + 1][val[1]].wall and matrix[val[0]][val[1] + 1].wall:
                    self.memo[box_pos] = float('inf')
                    return float('inf')
            else:
                targets_left -= 1
            total += self.buff[val[0]][val[1]]
        self.memo[box_pos] = total * box_moves * targets_left
        return total * box_moves * targets_left


def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection)
    else:
        problem = SokobanProblem(map, dead_detection)

    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    search = util.AStarSearch(heuristic=h)

    search.solve(problem)
    if search.actions is not None:
        f = open("temp.txt", "w")
        f.write("{}".format(search.actions))
        f.close()
    if 'f' in algorithm:
        return search.totalCost, search.actions, search.numStatesExplored
    else:
        return search.totalCost, search.actions, search.numStatesExplored


def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i + 1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)


def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + str(level):
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else:
                    break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')


def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels


def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    tic = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, algorithm, dead)
    toc = datetime.datetime.now()
    seq = ''.join(sol)
    if simulate:
        animate_sokoban_solution(map, seq)


def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="ucs | [f][a[2]] | c | all")

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if algorithm == 'c':
        algorithm = 'fa2'
        dead = True

    def solve_now():
        solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(maxSeconds):
        try:
            util.TimeoutFunction(solve_now, maxSeconds)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            # gc.collect()
            print('Memory limit exceeded.')
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % maxSeconds)

    solve_with_timeout(maxSeconds)


if __name__ == '__main__':
    main()
