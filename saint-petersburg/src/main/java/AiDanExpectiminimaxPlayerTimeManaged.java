import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

public class AiDanExpectiminimaxPlayerTimeManaged extends SPPlayer {

    int maxSearchDepth = 8; // Renamed from searchDepth
    AIDanSPStateFeaturesLR3 features = new AIDanSPStateFeaturesLR3();
    boolean verbose = false;
    boolean useTranspositionTable = true;
    int chanceSamples = 3; // Number of random outcomes to sample at chance nodes
    Random rng = new Random();

    // --- NEWLY ADDED TIME MANAGEMENT FIELDS ---
    // (Copied from AIDanSPMCTSPlayer)
    private long startMs = UNKNOWN_TIME; // start time of move computation
    int endEstimatePlayouts = 3; // number of playouts to estimate the
    //  number of remaining decisions
    double fOpening = 1.1; // bias towards opening move search
    // --- END OF NEW FIELDS ---


    public AiDanExpectiminimaxPlayerTimeManaged() {
        super("AiDanExpectiminimaxPlayerTimeManaged");
    }

    // Updated to set maxSearchDepth
    public void setSearchDepth(int depth) {
        this.maxSearchDepth = depth;
    }

    public void setUseTranspositionTable(boolean use) {
        this.useTranspositionTable = use;
    }
    
    public void setChanceSamples(int samples) {
        this.chanceSamples = samples;
    }

    @Override
    public int getAction(SPState state) {
        ArrayList<SPAction> actions = state.getLegalActions();
        if (actions.isEmpty()) return -1;

        int originalPlayer = state.playerTurn;
        if (verbose) System.out.println("Number of legal actions: " + actions.size());

        // --- NEW: TIME BUDGET CALCULATION ---
        startMs = System.currentTimeMillis();
        long turnSearchTimeMillis = 1000L; // default fixed time per move

        if (timeRemainingMillis != UNKNOWN_TIME) {
            // Estimate the number of decisions remaining (from MCTSPlayer)
            int currentPlayer = state.playerTurn;
            int totalDecisions = 0;
            for (int p = 0; p < endEstimatePlayouts; p++) {
                SPState simState = state.clone();
                while (!simState.isGameOver()) {
                    if (simState.playerTurn == currentPlayer) {
                        totalDecisions++;
                    }
                    ArrayList<SPAction> legalActions = simState.getLegalActions();
                    int actionIndex = (int) (Math.random() * legalActions.size());
                    SPAction action = legalActions.get(actionIndex);
                    simState = action.take();
                }
            }
            double movesExpected = (double) totalDecisions / endEstimatePlayouts;
            if (verbose) { // print the estimated number of decisions remaining
                System.out.printf("Estimated decisions remaining: %.2f\n", movesExpected);
            }
            // Allocate time for this move
            turnSearchTimeMillis = (long) (fOpening * timeRemainingMillis / movesExpected);
            // Ensure that the move time is not more than a 20th of the remaining time
            turnSearchTimeMillis = Math.min(turnSearchTimeMillis,
                    timeRemainingMillis / 20L);
        }
        // --- END: TIME BUDGET CALCULATION ---

        // Clear stats, but not the table (TT helps iterative deepening)
        resetStats();
        
        int bestActionIndex = 0; // Default to first action
        double bestValueSoFar = Double.NEGATIVE_INFINITY;

        // --- NEW: Iterative Deepening Loop ---
        for (int currentDepth = 1; currentDepth <= maxSearchDepth; currentDepth++) {
            long depthStartTime = System.currentTimeMillis();
            
            double bestValueThisDepth = Double.NEGATIVE_INFINITY;
            int bestActionIndexThisDepth = 0;
            
            double alpha = Double.NEGATIVE_INFINITY;
            double beta = Double.POSITIVE_INFINITY;

            // Root: always maximizing player's turn
            for (int i = 0; i < actions.size(); i++) {
                SPState copy = state.clone();
                ArrayList<SPAction> clonedActions = copy.getLegalActions();
                SPAction actionToTake = clonedActions.get(i);
                
                actionToTake.take(); 
                
                // Alpha-beta with chance node sampling
                double value = alphaBeta(copy, currentDepth - 1, originalPlayer, alpha, beta);

                if (verbose && currentDepth == maxSearchDepth) { // Only print for final depth
                    System.out.printf("  Action %s evaluated to %.4f\n", actions.get(i), value);
                }

                if (value > bestValueThisDepth) {
                    bestValueThisDepth = value;
                    bestActionIndexThisDepth = i;
                }
                
                alpha = Math.max(alpha, bestValueThisDepth);
            }
            
            // Store the result from this completed depth
            bestActionIndex = bestActionIndexThisDepth;
            bestValueSoFar = bestValueThisDepth;

            if (verbose) {
                System.out.printf("Depth %d complete. Best action: %s (Value %.4f)\n",
                    currentDepth, actions.get(bestActionIndex), bestValueSoFar);
            }

            // --- NEW: Time Check ---
            long depthTimeTaken = System.currentTimeMillis() - depthStartTime;
            long totalElapsed = System.currentTimeMillis() - startMs;

            // Heuristic: If (total time + 2*last_depth_time) > budget,
            // we probably don't have time for the next (much longer) depth.
            if (totalElapsed + (depthTimeTaken * 2) > turnSearchTimeMillis && currentDepth < maxSearchDepth) {
                if (verbose) {
                    System.out.printf("Stopping at depth %d (took %dms, total %dms) due to time limit.\n",
                        currentDepth, depthTimeTaken, totalElapsed);
                }
                break; // Stop iterative deepening
            }
            
            // Hard cutoff if we are already over time
            if (totalElapsed > turnSearchTimeMillis && currentDepth < maxSearchDepth) {
                 if (verbose) {
                    System.out.printf("Stopping at depth %d (took %dms, total %dms) due to HARD time limit.\n",
                        currentDepth, depthTimeTaken, totalElapsed);
                }
                break;
            }
            // --- End Time Check ---
        }

        if (verbose) {
            System.out.printf("Chosen Action: %s (Value %.4f)\n", actions.get(bestActionIndex), bestValueSoFar);
            System.out.printf("TT Stats: Size=%d, Hits=%d, Stores=%d\n", 
                transTable.size(), ttHits, ttStores);
        }
        
        return bestActionIndex;
    }

    /**
     * Alpha-beta minimax with Monte Carlo sampling at chance nodes.
     * When random events occur (like Observatory draws), samples a few outcomes
     * and averages them instead of assuming a single deterministic outcome.
     */
    private double alphaBeta(SPState state, int depth, int originalPlayer, double alpha, double beta) {
        long hash = 0;
        
        // Compute Zobrist hash if using TT
        if (useTranspositionTable) {
            hash = AiDanZHashing.computeHash(state);
            
            // Check transposition table
            TTEntry entry = transTable.get(hash);
            // *** CRUCIAL FOR IDDFS: only use entry if it was from a
            // search at least as deep as the one we need now. ***
            if (entry != null && entry.depth >= depth) {
                ttHits++;
                
                // Exact value
                if (entry.lowerBound == entry.upperBound) {
                    return entry.lowerBound;
                }
                
                // Use bounds to prune
                if (entry.lowerBound >= beta) {
                    return entry.lowerBound;
                }
                if (entry.upperBound <= alpha) {
                    return entry.upperBound;
                }
                
                // Narrow the search window
                alpha = Math.max(alpha, entry.lowerBound);
                beta = Math.min(beta, entry.upperBound);
            }
        }
        
        // Base case
        if (depth == 0 || state.isGameOver()) {
            double rawValue = features.predict(state);
            // Adjust value if it's not the original player's turn to move
            double value = (state.playerTurn == originalPlayer || state.isGameOver()) 
                ? rawValue : 1.0 - rawValue;
            
            if (useTranspositionTable) {
                // Store with the exact depth
                transTable.put(hash, new TTEntry(value, value, depth));
                ttStores++;
            }
            
            return value;
        }

        ArrayList<SPAction> actions = state.getLegalActions();
        if (actions.isEmpty()) {
            // No legal actions, treat as base case
            double rawValue = features.predict(state);
             double value = (state.playerTurn == originalPlayer || state.isGameOver()) 
                ? rawValue : 1.0 - rawValue;
            
            if (useTranspositionTable) {
                transTable.put(hash, new TTEntry(value, value, depth));
                ttStores++;
            }
            return value;
        }

        boolean maximizing = (state.playerTurn == originalPlayer);
        double origAlpha = alpha;
        double origBeta = beta;
        
        if (maximizing) {
            double maxEval = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < actions.size(); i++) {
                // Sample multiple random outcomes for this action
                double sampledValue = sampleAction(state, i, depth, originalPlayer, alpha, beta);
                
                maxEval = Math.max(maxEval, sampledValue);
                alpha = Math.max(alpha, maxEval);
                
                if (beta <= alpha) break;
            }
            
            if (useTranspositionTable) {
                // Store bounds based on alpha-beta results
                double lower = maxEval;
                double upper = (maxEval >= origBeta) ? Double.POSITIVE_INFINITY : maxEval;
                
                if (maxEval <= origAlpha) {
                    upper = maxEval;
                    lower = Double.NEGATIVE_INFINITY;
                }
                
                transTable.put(hash, new TTEntry(lower, upper, depth));
                ttStores++;
            }
            
            return maxEval;
        } else { // Minimizing
            double minEval = Double.POSITIVE_INFINITY;
            for (int i = 0; i < actions.size(); i++) {
                // Sample multiple random outcomes for this action
                double sampledValue = sampleAction(state, i, depth, originalPlayer, alpha, beta);
                
                minEval = Math.min(minEval, sampledValue);
                beta = Math.min(beta, minEval);
                
                if (beta <= alpha) break;
            }
            
            if (useTranspositionTable) {
                // Store bounds based on alpha-beta results
                double upper = minEval;
                double lower = (minEval <= origAlpha) ? Double.NEGATIVE_INFINITY : minEval;
                
                if (minEval >= origBeta) {
                    lower = minEval;
                    upper = Double.POSITIVE_INFINITY;
                }
                
                transTable.put(hash, new TTEntry(lower, upper, depth));
                ttStores++;
            }
            
            return minEval;
        }
    }
    
    /**
     * Sample an action multiple times with different random seeds to handle randomness.
     * Returns the average value across samples.
     */
    private double sampleAction(SPState state, int actionIndex, int depth, 
                                int originalPlayer, double alpha, double beta) {
        double totalValue = 0.0;
        
        for (int sample = 0; sample < chanceSamples; sample++) {
            SPState copy = state.clone();
            
            // Shuffle decks slightly to simulate randomness in draws
            if (sample > 0) { // First sample uses original ordering
                Collections.shuffle(copy.workerDeck, rng);
                Collections.shuffle(copy.buildingDeck, rng);
                Collections.shuffle(copy.aristocratDeck, rng);
                Collections.shuffle(copy.tradingDeck, rng);
            }
            
            ArrayList<SPAction> clonedActions = copy.getLegalActions();
            clonedActions.get(actionIndex).take();
            
            double value = alphaBeta(copy, depth - 1, originalPlayer, alpha, beta);
            totalValue += value;
        }
        
        return totalValue / chanceSamples;
    }

    // --- Transposition Table and Stats ---
    
    public static class TTEntry {
        double lowerBound, upperBound;
        int depth; // Store the depth this entry was calculated at

        TTEntry(double lb, double ub, int d) { 
            lowerBound = lb; 
            upperBound = ub; 
            depth = d; 
        }
    }

    // Using LRU cache for TT
    private Map<Long, TTEntry> transTable = new LinkedHashMap<>(1000000, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<Long, TTEntry> eldest) {
            return size() > 1000000;
        }
    };
    
    private long ttHits = 0;
    private long ttStores = 0;
    
    public void resetStats() {
        ttHits = 0;
        ttStores = 0;
    }
    
    public void clearTranspositionTable() {
        transTable.clear();
        resetStats();
    }
    
    public String getTTStats() {
        return String.format("TT: Size=%d, Hits=%d, Stores=%d, Hit Rate=%.2f%%", 
            transTable.size(), ttHits, ttStores, 
            ttStores > 0 ? (100.0 * ttHits / (ttStores + ttHits)) : 0.0); // More accurate rate
    }
    
}