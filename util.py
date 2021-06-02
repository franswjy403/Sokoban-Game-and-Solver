import heapq, collections, os, signal, datetime, random

class SearchProblem:
    def start(self): raise NotImplementedError("Override me")

    def goalp(self, state): raise NotImplementedError("Override me")

    def expand(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    def solve(self, problem): raise NotImplementedError("Override me")

class AStarSearch(SearchAlgorithm):

    def __init__(self, heuristic, verbose=0):
        self.verbose = verbose
        self.heuristic = heuristic

    def solve(self, problem):
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0
        self.finalCosts = collections.defaultdict(lambda:float('inf'))

        frontier = PriorityQueue()
        backpointers = {}

        startState = problem.start()
        self.finalCosts[startState] = 0
        frontier.update(startState, self.heuristic(startState))
        estCostNotified = 0

        while True:
            state, estimatedCost = frontier.removeMin()

            if state == None: break

            pastCost = self.finalCosts[state]

            if (self.verbose >= 1 and estimatedCost > estCostNotified) or (self.verbose >= 2 and random.randint(0,1000)==0):
                print('estimatedCost {} started, {} states expanded, sample state is {}'.format(estimatedCost, self.numStatesExplored, state.data))
                if self.verbose >= 2:
                    print('h value is {}'.format(self.heuristic(state)))
                    while state != startState:
                        action, prevState = backpointers[state]
                        print('   from action {}'.format(action))
                        state = prevState
                estCostNotified = estimatedCost

            self.numStatesExplored += 1
            if self.verbose >= 2:
                print(("Exploring %s with pastCost %s and estimated cost %s" % (state, pastCost, estimatedCost)))

            if problem.goalp(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print(("numStatesExplored = %d" % self.numStatesExplored))
                    print(("totalCost = %s" % self.totalCost))
                    print(("actions = %s" % self.actions))
                return

            for action, newState, cost in problem.expand(state):
                if self.verbose >= 3:
                    print(("  Action %s => %s with cost %s + %s + %s" % (action, newState, pastCost, cost, estimatedCost)))
                newPastCost = pastCost + cost
                self.finalCosts[newState] = min(newPastCost,self.finalCosts[newState])

                if frontier.update(newState, newPastCost + self.heuristic(newState)):
                    backpointers[newState] = (action, state)
        if self.verbose >= 1:
            print("No path found")

class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {} 
    
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) 


class TimeoutFunctionException(Exception):
    pass
class TimeoutFunction:
    def __init__(self, function, maxSeconds):
        self.maxSeconds = maxSeconds
        self.function = function

    def handle_maxSeconds(self, signum, frame):
        print('TIMEOUT!')
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if os.name == 'nt':
            
            timeStart = datetime.datetime.now()
            result = self.function(*args)
            timeEnd = datetime.datetime.now()
            if timeEnd - timeStart > datetime.timedelta(seconds=self.maxSeconds + 1):
                raise TimeoutFunctionException()
            return result
            
        old = signal.signal(signal.SIGALRM, self.handle_maxSeconds)
        signal.alarm(self.maxSeconds + 1)
        result = self.function(*args)
        signal.alarm(0)
        return result
