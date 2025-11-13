import java.util.ArrayList;

public class AIDanSPPlayerFMCTrainerTimeManaged extends SPPlayer {

    int numSimulationsPerAction = 10000; // Acts as MAX simulations per action
    int playoutTerminationDepth = 4;
    AIDanSPStateFeaturesLR3 features = new AIDanSPStateFeaturesLR3();
    boolean verbose = false;

    // --- TIME MANAGEMENT FIELDS ---
    private long startMs = UNKNOWN_TIME; // start time of move computation
    int endEstimatePlayouts = 3; // number of playouts to estimate the
    //  number of remaining decisions
    double fOpening = 1.1; // bias towards opening move search
    int numBlockIterations = 10; // Check time every 10 simulation rounds

    public AIDanSPPlayerFMCTrainerTimeManaged() {
        super("AIDanSPPlayerFMCTrainerTimeManaged");
    }

    @Override
    public int getAction(SPState state) {
        ArrayList<SPAction> actions = state.getLegalActions();
        int numActions = actions.size();
        if (numActions == 0) return -1;
        if (numActions == 1) return 0;

        if (verbose)
            System.out.println("Number of legal actions: " + numActions);

        startMs = System.currentTimeMillis();
        long turnSearchTimeMillis = 1000L;

        if (timeRemainingMillis != UNKNOWN_TIME) {
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
            if (verbose) {
                System.out.printf("Estimated decisions remaining: %.2f\n", movesExpected);
            }
            turnSearchTimeMillis = (long) (fOpening * timeRemainingMillis / movesExpected);
            turnSearchTimeMillis = Math.min(turnSearchTimeMillis,
                    timeRemainingMillis / 20L);
        }
        
        double[] actionTotalValues = new double[numActions];
        int simsPerActionCount = 0;
        long blockStartMillis = System.currentTimeMillis();

        for (int j = 0; j < numSimulationsPerAction; j++) {

            if (j % numBlockIterations == 0) {
                long currentMillis = System.currentTimeMillis();
                long lastIterationBlockMillis = currentMillis - blockStartMillis;
                long elapsedMillis = currentMillis - startMs;

                if (elapsedMillis + lastIterationBlockMillis > turnSearchTimeMillis) {
                    if (verbose) {
                        System.out.printf("FMC terminating at %d simulations per action due to time limit.\n", j);
                    }
                    break;
                }
                blockStartMillis = currentMillis;
            }

            for (int i = 0; i < numActions; i++) {
                SPState depth1Copy = state.clone();
                SPAction action = depth1Copy.getLegalActions().get(i);
                action.take();

                SPState simCopy = depth1Copy.clone();

                for (int k = 0; !simCopy.isGameOver() && k < playoutTerminationDepth; k++) {
                    ArrayList<SPAction> legalActions = simCopy.getLegalActions();
                    SPAction randomAction = legalActions.get((int) (Math.random() * legalActions.size()));
                    randomAction.take();
                }

                int scoreDiff = simCopy.playerPoints[state.playerTurn] - simCopy.playerPoints[1 - state.playerTurn];
                double heuristicValue = eval(simCopy) + 0.005 * scoreDiff;
                if (state.playerTurn != simCopy.playerTurn) {
                    heuristicValue = 1 - heuristicValue;
                }
                
                actionTotalValues[i] += heuristicValue;
            }
            
            simsPerActionCount++;
        }
        
        // If time ran out before even one round, pick randomly
        if (simsPerActionCount == 0) {
            if (verbose) System.out.println("FMC: No simulations completed, picking random action.");
            return (int) (Math.random() * numActions);
        }

        int bestActionIndex = -1;
        double bestValue = Double.NEGATIVE_INFINITY;

        // Find the action with the best average value
        for (int i = 0; i < numActions; i++) {
            double estValue = actionTotalValues[i] / simsPerActionCount;
            
            if (verbose) {
                System.out.printf("  Action %d (%s): Avg Value %.4f (Sims: %d)\n",
                        i, actions.get(i).toString(), estValue, simsPerActionCount);
            }

            if (estValue > bestValue) {
                bestValue = estValue;
                bestActionIndex = i;
            }
        }

        if (verbose)
            System.out.printf("%s selected (est. value %.4f)\n", actions.get(bestActionIndex), bestValue);
            
        return bestActionIndex;
    }

    private double eval(SPState state) {
        return features.predict(state);
    }
}