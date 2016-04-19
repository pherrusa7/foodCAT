
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    #We use a stack for the dfs algorithm
    frontier= util.Stack()
    return genericSearch(problem, frontier, nullHeuristic)

    
#The only difference between all the search algorithms to implement is the frontier and the heuristic used, in the case of dfs and bfs 
#we defined a null heuristic on the function nullHeuristic. 
def genericSearch(problem, frontier, heuristic):

  #For every node (grid position) we store the cost, the parent (by that we mean the node we came from) and the direction we came from.
  #We use a dictionary for that. We use the node(the grid position) as a key.
  #Key: Node. Value: Cost , parent, direction
  #We initialize the dict with the node we start, we use (-1,-1) since it has no parent, and has no direction we came from.
  visitedNodes = {problem.getStartState(): (0, (-1,-1), None)}
  
  #Add the start to the frontier, we also add the cost, combined cost in case of A*.
  frontier.push((problem.getStartState(), 0))
  goalState = None
  goalStateFound= False
  #we start looking the nodes until we are done.
  while not goalStateFound and not frontier.isEmpty():
    #get the node from the frontier, we only want the position. On the frontier we also got the cost (combined with heuristic if we have one).
    currentNode = frontier.pop()[0]
    #get the cost, the parent and the direction of the current node
    currentNodecost = visitedNodes[currentNode][0]
    currentNodeParent = visitedNodes[currentNode][1]
    currentNodeDirection= visitedNodes[currentNode][2]
    #get the neighbors of the current node
    neighbors = problem.getSuccessors(currentNode)
    #Add to the frontier the neighbors of the current node
    for node in neighbors:
      #we only explore the node if it's not already visited.
      #Node[0] is the gridPosition, remember we use that as the key of the dict.
      if node[0] not in visitedNodes:
        newParent = currentNode
        newDirection = node[1]
        #We say newCost but we mean cost to the path plus the value of the heuristic
        newCost=currentNodecost+node[2]+heuristic(node[0], problem)
        currentNodeData = (newCost, newParent, newDirection)
        # Add to dict 'VisitedNodes'the node (the neighbour) with the data we need. 
        visitedNodes[node[0]] = currentNodeData
        # Add to frontier the currrentNode (the neighbour)
        frontier.push((node[0], newCost))
        #Check if that neighbour is the goal.
        if problem.isGoalState(node[0]):
          goalStateFound=True
          goalState=node[0]
          break
        
  #Only return a path if we found the goal  
  if goalStateFound:
    path=[]
    while goalState!=problem.getStartState():
        path.append(visitedNodes[goalState][2])
        goalState=visitedNodes[goalState][1]
    path.reverse()
    return path        
  else:
    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    #We use a queue for the bfs algorithm
    frontier= util.Queue()
    return genericSearch(problem, frontier, nullHeuristic)
    

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    #We use this heuristic on the bfs and dfs.
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
  #We use a priority queue for the A star.
  frontier = util.PriorityQueueWithFunction(priorityfunction)
  return genericSearch(problem, frontier, heuristic)

    
def priorityfunction(item):
  return item[1]

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
