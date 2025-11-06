import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Comparator;

public class SPMCTSPlayer extends SPPlayer { // simplified and ported from
    // Marc Lanctot's OpenSpiel MCTS implementation in C++

    double uctC = 2.0;                                          // UCT exploration constant
    int numChanceSamples = 10;                                  // number of chance samples per chance node
    int numIterations = 20000;                                  // number of MCTS iterations per move
    int playoutTerminationDepth = 4;                            // depth to terminate playouts
    SPStateFeaturesLR1 features = new SPStateFeaturesLR1();     // state features for playout policy
    boolean verbose = true;                                     // Verbosity flag
    Random chanceSeedRng = new Random(12345);              // RNG for chance sampling
    int nodes = 0;

    public SPMCTSPlayer() {
        super("SPMCTSPlayer");
    }

    class SearchNode {
        public int action = 0; // action index leading to this node
        public int player = 0; // player that acted to reach this node, or a negative seed value for a chance node

        public int exploreCount = 0; // number of times node has been explored
        public double totalReward = 0.0; // total reward from simulations through this node
        public final List<SearchNode> children = new ArrayList<>(); // child nodes

        public SearchNode(int action, int player) {
            this.action = action;
            this.player = player;
        }

        public double uctValue(double parentExploreCount, double uctC) {
            if (exploreCount == 0) {
                return Double.POSITIVE_INFINITY; // ensure unvisited nodes are explored
            }
            double exploitation = totalReward / exploreCount;
            double exploration = uctC * Math.sqrt(Math.log(parentExploreCount) / exploreCount);
            return exploitation + exploration;
        }

        public SearchNode bestChildByVisits() {
            return children.stream()
                    .max((a, b) -> Integer.compare(a.exploreCount, b.exploreCount))
                    .orElse(null);
        }

        public SearchNode bestChildUCT() {
            return children.stream()
                    .max((a, b) -> Double.compare(a.uctValue(this.exploreCount, uctC), b.uctValue(this.exploreCount, uctC)))
                    .orElse(null);
        }

        public String toString() {
            return String.format("%d: Player: %d, Visits: %d, Q: %.2f, Children: %d", 
            action, player, exploreCount,
            (exploreCount == 0) ? 0.0 : totalReward / exploreCount, 
            children.size());
        }

        public String childrenStr(int parentExploreCount, double uctC) {
            StringBuilder sb = new StringBuilder();
            Comparator<SearchNode> comparator = (a, b) -> Double.compare(
                a.uctValue(parentExploreCount, uctC), 
                b.uctValue(parentExploreCount, uctC)
            );
            children.stream()
                .sorted(comparator.reversed())
                .forEach(child -> {
                    sb.append(String.format("%s\n", child.toString()));
                });

            return sb.toString();
        }

        public boolean isChanceNode() {
            return player < 0;
        }
    }

    
    // count of nodes created 
    @Override
    public int getAction(SPState state) {
        // get the legal actions for the current state,
        // compute the number of legal actions,
        // call MCTSearch to get the root SearchNode,
        // select the best child of the root node by visit count,
        // return the action index of that best child.
        ArrayList<SPAction> legalActions = state.getLegalActions();
        int numActions = legalActions.size();
        SearchNode root = MCTSearch(state);
        SearchNode bestChild = root.bestChildByVisits();
        if (verbose) {
            System.out.println("Number of legal actions: " + numActions);
            System.out.println("Root node children:");
            System.out.println(root.childrenStr(root.exploreCount, uctC));
            System.out.println("Selected action: " + legalActions.get(bestChild.action));
        }

        return bestChild.action;
    }

    public SearchNode MCTSearch(SPState rootState) {
        SearchNode rootNode = new SearchNode(0, rootState.playerTurn);
        expand(rootNode, rootState);
        nodes = 1; // reset node counter
        long startMillis = System.currentTimeMillis();

        List<SearchNode> path = new ArrayList<>(); // store sequence of SearchNodes visited
        for (int iter = 0; iter < numIterations; iter++) {
            path.clear();
            SPState state = rootState.clone();
            SearchNode node = rootNode;

            // Selection/Expansion phase (TREE_POLICY)
            // While the state is non-terminal:
            // - add the non-chance node to the path
            // - If the nodes is not fully expanded, expand it and break
            // - Otherwise, select the best child by UCT, 
            //   (prioritizing unselected children) 
            // - If the action is a chance actiom expand the chance nodes
            //   if needed, and reproduce a chance-sampled outcome using
            //   the stored seed + action index.
            // We assume that chance nodes are not followed by chance nodes,
            // After this phase, the path will contain all non-teminal nodes visited,
            //   and we will either be at a terminal state or have just left the search tree.
            while (!state.isGameOver()) {
                path.add(node);
                if (node.children.isEmpty()) {
                    // Node is not expanded
                    expand(node, state); // expand and break
                    break;
                }
                // Node is expanded; select best child by UCT
                SearchNode nextNode = node.bestChildUCT();
                if (nextNode.isChanceNode()) {
                    path.add(nextNode); // add chance node to path
                    // Chance node: reproduce chance outcome
                    SPAction chanceAction = state.getLegalActions().get(nextNode.action);
                    // Use stored seed + action index to seed RNG
                    int sampleIndex = (int) (Math.random() * numChanceSamples);
                    int sampleSeed = -nextNode.player + sampleIndex;
                    state = chanceAction.take(sampleSeed);
                    node = nextNode.children.get(sampleIndex);
                } else {
                    // Regular action node
                    // TODO
                }
            }

            // Expansion
            if (!state.isGameOver()) {
                expand(node, state);
                // Move to one of the new children
                if (!node.children.isEmpty()) {
                    node = node.children.get(0); // could randomize or use UCT here
                    ArrayList<SPAction> legalActions = state.getLegalActions();
                    SPAction action = legalActions.get(node.action);
                    action.take();
                    path.add(node);
                }
            }

            // Simulation
            int depth = 0;
            while (!state.isGameOver() && depth < playoutTerminationDepth) {
                ArrayList<SPAction> legalActions = state.getLegalActions();
                SPAction randomAction = legalActions.get((int) (Math.random() * legalActions.size()));
                randomAction.take();
                depth++;
            }
            double reward = features.predict(state);

            // Backpropagation
            for (SearchNode n : path) {
                n.exploreCount += 1;
                n.totalReward += reward;
            }
        }



        if (verbose) {
            System.out.println("Total nodes created during MCTS: " + nodes);
            System.out.println("Total seconds for MCTS: " + ((System.currentTimeMillis() - startMillis) % 100000) / 1000.0);
        }

        return rootNode;
    }

    public void expand(SearchNode node, SPState state) {
        // Assumes state is non-terminal, non-chance
        // For each non-chance legal action in state, add a child SearchNode.
        // For each chacne action, create an expanded chance sampling SearchNode
        //    with a negative "player" value for reproducable chance sampling, and children that reflect the player
        //    that chose the chance action
        ArrayList<SPAction> legalActions = state.getLegalActions();

        // Create a random permuation of the action indices in int array actionIndices
        int numActions = legalActions.size();
        int[] actionIndices = new int[numActions];
        for (int i = 0; i < numActions; i++) {
            actionIndices[i] = i;
        }

        // Fisher-Yates shuffle (really Durstenfeld's shuffle 1994)
        // and later Knuth shuffle (1969)
        for (int i = 0; i < numActions; i++) {
            int j = (int) (Math.random() * (i + 1));
            int temp = actionIndices[i];
            actionIndices[i] = actionIndices[j];
            actionIndices[j] = temp;
        }

        for (int a : actionIndices) {
            SPAction action = legalActions.get(a);
            int player = state.playerTurn;
            if (action.isChanceAction()) {
                // Create chance sampling node
                int chanceSeed = -(chanceSeedRng.nextInt(Integer.MAX_VALUE - numChanceSamples));
                SearchNode chanceNode = new SearchNode(a, chanceSeed);
                node.children.add(chanceNode);
                nodes++;

                // Create children for chance node
                for (int sample = 0; sample < numChanceSamples; sample++) {
                    SearchNode childNode = new SearchNode(a, player);
                    chanceNode.children.add(childNode);
                    nodes++;
                }
            } else {
                SearchNode childNode = new SearchNode(a, player);
                node.children.add(childNode);
                nodes++;
            }
        }
        
    }

}
