import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import ai.catboost.CatBoostError;
import ai.catboost.CatBoostModel;
import ai.catboost.CatBoostPredictions;
import smile.classification.RandomForest;
import smile.classification.RandomForest.Options;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.io.Read;

public class AIDanSPStateFeaturesRF1 {
    String modelFilename = "AIDanSPRandomForest.model";
    // LogisticRegression.Binomial model;
    private RandomForest model;
    ArrayList<SPFeature> features;
    ArrayList<SPFeature> rlFeatures;
    // private SharedInterpreter interp;
    // private boolean cbInitialized = false;
    // ADD THESE
    private CatBoostModel roundsLeftModel;
    String rlModelFilename = "rounds_left_model.cbm";

    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public ArrayList<String> getFeatureNames() {
        ArrayList<String> names = new ArrayList<>();
        for (SPFeature feature : features) {
            names.add(feature.getName());
        }
        return names;
    }

    public ArrayList<Object> getRLTrainFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : rlFeatures) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public ArrayList<String> getRLTrainFeature () {
        ArrayList<String> result = new ArrayList<>();
        for (SPFeature feature : rlFeatures) {
            result.add(feature.getName());
        }
        return result;
    }

    public AIDanSPStateFeaturesRF1() {
        // features = new ArrayList<>();
        // features.add(new SPFeatureMinDeckSize());
        // features.add(new SPFeaturePoints());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRubles());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureUniqueAristocratsPointsDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsPointsDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsRoundGain());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsRoundGainDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesRoundGain());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesRoundGainDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureCardsInHand());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureCardsInHandDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureHandSpaceDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureHandSpaceDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureBuyableCardsInHand());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureBuyableCardsInHand(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureDupAristoCount());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureDupAristoCount(), new SPFeatureMinDeckSize()));

        features = new ArrayList<>();
        features.add(new SPFeatureRoundsLeft());
        features.add(new SPFeaturePoints());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureRoundsLeft()));
        features.add(new SPFeaturePointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureRubles());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureRublesDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureUniqueAristocratsPointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsPointsDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeaturePointsRoundGain());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureRoundsLeft()));
        features.add(new SPFeaturePointsRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureRublesRoundGain());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureRublesRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureCardsInHand());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureNumLegalMoves());
        features.add(new SPFeatureInteractionTerm(new SPFeatureNumLegalMoves(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureCardsInHandDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureHandSpaceDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureHandSpaceDiff(), new SPFeatureRoundsLeft()));

        // features = new ArrayList<>();
        // features.add(new SPFeatureMinDeckSize());
        // // features.add(new SPFeaturePoints());
        // // features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureRubles());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureUniqueAristocratsPointsDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsPointsDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeaturePointsRoundGain());
        // // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsRoundGainDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesRoundGain());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesRoundGainDiff());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureCardsInHand());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureCardsInHandDiff());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureHandSpaceDiff());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureHandSpaceDiff(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureBuyableCardsInHand());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureBuyableCardsInHand(), new SPFeatureMinDeckSize()));
        // // features.add(new SPFeatureDupAristoCount());
        // // features.add(new SPFeatureInteractionTerm(new SPFeatureDupAristoCount(), new SPFeatureMinDeckSize()));

        // Initialize features list for rounds left estimator
        rlFeatures = new ArrayList();
        rlFeatures.add(new SPFeatureCurrentRound());
        for (int phase = 0; phase < 4; phase++) {
            rlFeatures.add(new SPFeatureCardsInDeck(phase));
            if (phase < 3) 
                for (int player = 0; player < 2; player++) {
                    rlFeatures.add(new SPFeaturePointsPerPhase(phase, player));
                    rlFeatures.add(new SPFeatureRublesPerPhase(phase, player));
                }
        }
        rlFeatures.add(new SPFeatureMinDeckSize());
        for (int player = 0; player < 2; player++) {
            rlFeatures.add(new SPFeaturePlayerNPoints(player));
            rlFeatures.add(new SPFeaturePlayerNRubles(player));
            rlFeatures.add(new SPFeatureUniqueAristocratPoints(player));
            rlFeatures.add(new SPFeatureDuplicateAristocrats(player));
            rlFeatures.add(new SPFeatureCardsInHandN(player));
        }
        rlFeatures.add(new SPFeatureCardsOfferedTop());
        rlFeatures.add(new SPFeatureCardsOfferedBottom());

        // cbInitialized = initPythonCB("rounds_remaining.csv");
        initializeRoundsLeftModel();
        initializeModel();
    }

    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
        }
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(
                new java.io.FileInputStream(modelFilename))) {
            model = (RandomForest) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public String getCSVHeader() {
        StringBuilder header = new StringBuilder();
        for (SPFeature feature : features) {
            header.append(feature.getName()).append(",");
        }
        header.append("is_winner");
        return header.toString();
    }

    public String getCSVRow(SPState state, boolean[] isWinner, int numRounds) {
        int currentPlayerIndex = state.playerTurn;
        int winnerVal = isWinner[currentPlayerIndex] ? 1 : 0;

        // The "actual" rounds_left value
        int currPhase = (state.phase < 4) ? state.phase : (state.phase == 4) ? 2 : 0;
        double round = (double) state.round + (currPhase % 4) / 4.0;
        double rounds_left = numRounds - round;

        StringBuilder row = new StringBuilder();

        for (SPFeature feature : features) {
            Object valueToAppend;

            // If it's an interaction term involving "rounds_left"
            if (feature instanceof SPFeatureInteractionTerm) {
                SPFeatureInteractionTerm interactionFeature = (SPFeatureInteractionTerm) feature;
                
                SPFeature otherFeature = interactionFeature.getOtherFeatureIfRoundsLeft();

                if (otherFeature != null) {

                    Object otherValue = otherFeature.getValue(state);
                    
                    // Calculate the product using the *actual* rounds_left
                    if (otherValue instanceof Number) {
                        valueToAppend = ((Number) otherValue).doubleValue() * rounds_left;
                    } else {
                        valueToAppend = 0.0;
                    }
                } else {
                    // It's a different interaction term, use default behavior
                    valueToAppend = feature.getValue(state);
                }
            } 
            else if (feature.getName().equals("rounds_left")) {
                valueToAppend = rounds_left;
            } 
            else {
                valueToAppend = feature.getValue(state);
            }
            
            row.append(valueToAppend).append(",");
        }

        row.append(winnerVal);
        return row.toString();
    }

    public String getCSVRows(SPGameTranscript transcript) {
        StringBuilder rows = new StringBuilder();
        boolean[] isWinner = transcript.getWinners();
        int numRounds = transcript.getNumRounds();
        for (SPState state : transcript.getStates()) {
            rows.append(getCSVRow(state, isWinner, numRounds)).append("\n");
        }
        return rows.toString();
    }

    public void generateCSVData(String filename, int numGames) {
        int numThreads = 10;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        CompletionService<String> completionService = new ExecutorCompletionService<>(executor);
        
        AtomicInteger gamesCompleted = new AtomicInteger(0);
    
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println(getCSVHeader());
    
            for (int i = 0; i < numGames; i++) {
                completionService.submit(() -> {
                    // SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
                    SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPPlayerFlatMC(), new SPPlayerFlatMC());
                    return getCSVRows(transcript); 
                });
            }
    
            for (int i = 0; i < numGames; i++) {
                try {
                    Future<String> future = completionService.take(); 
                    
                    String csvRow = future.get(); 
                    
                    writer.print(csvRow);
    
                    int done = gamesCompleted.incrementAndGet();
                    System.out.println("Generated game " + done + "/" + numGames);
    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt(); 
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    System.err.println("Error during game simulation: " + e.getCause());
                    e.printStackTrace();
                }
            }
    
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown(); 
            try {
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow(); 
                Thread.currentThread().interrupt();
            }
        }
    }

    public String getRLCSVHeader() {
        StringBuilder header = new StringBuilder();
        ArrayList<String> rlFeatureCols = getRLTrainFeature();
        for (String feature : rlFeatureCols) {
            header.append(feature).append(",");
        }
        header.append("rounds_left");
        return header.toString();
    }

    public void generateRLCSVData(String filename, int numGames) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            // Write header
            writer.println(getRLCSVHeader());

            for (int i = 0; i < numGames; i++) {
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
                boolean[] isWinner = transcript.getWinners();
                int numRounds = transcript.getNumRounds();
                for (SPState state : transcript.getStates()) {
                    ArrayList<Object> featureValues = getRLTrainFeatureValues(state);
                    StringBuilder row = new StringBuilder();

                    for (Object value : featureValues) {
                        row.append(value).append(",");
                    }

                    int currPhase = (state.phase < 4) ? state.phase : (state.phase == 4) ? 2 : 0;
                    double round = (double) state.round + (currPhase % 4) / 4.0;
                    double rounds_left = numRounds - round;
                    row.append(rounds_left);
                    writer.println(row.toString());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void initializeRoundsLeftModel() {
        File modelFile = new File(rlModelFilename);
        if (!modelFile.exists()) {
            System.err.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            System.err.println("Rounds Left Model file not found: " + rlModelFilename);
            System.err.println("Please train the model using Python and place it in the project root.");
            System.err.println("You can generate training data by calling generateRLCSVData()");
            System.err.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            roundsLeftModel = null;
        } else {
            try {
                roundsLeftModel = CatBoostModel.loadModel(rlModelFilename);
                System.out.println("Rounds Left CatBoost model loaded successfully.");
            } catch (CatBoostError e) {
                System.err.println("CatBoost model loading failed:");
                e.printStackTrace();
                roundsLeftModel = null;
            }
        }
    }

    public void learnModel() {

        try {
            String trainingDataFile = "AIDanSPTrainingData.csv";
            int numGames = 1000; // Number of games to simulate for training data
            generateCSVData(trainingDataFile, numGames);
            DataFrame df = Read.csv(trainingDataFile, "header=true");

            df = df.factorize("is_winner");

            Formula formula = Formula.lhs("is_winner");

            System.out.println(formula);

            int ntrees = 200;
            int mtry = (int) df.ncol()/3;
            int maxDepth = 100;
            int maxNodes = 100;
            int nodeSize = 5;

            // Configure options
            Options opts = new Options(ntrees, mtry, maxDepth, maxNodes, nodeSize);

            // Fit model
            RandomForest model = RandomForest.fit(formula, df, opts);

            System.out.println("RandomForest model trained and saved. Features: " + (df.ncol() - 1));

            // Save model
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilename))) {
                oos.writeObject(model);
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
    }

    public double predict(SPState state) {
        if (model == null) {
            throw new IllegalStateException("Model has not been trained or loaded.");
        }

        // Build feature vector
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }

        String[] featureNames = getFeatureNames().toArray(new String[0]);
        Tuple testInstance = toTuple(featureValues, featureNames);

        // Get posterior probabilities via voting
        double[] posteriori = new double[2];
        int predictedClass = model.predict(testInstance, posteriori); // fills posteriori

        // posteriori[0] = P(class 0), posteriori[1] = P(class 1)
        // Return probability that is_winner == class 1
        double probIsWinner = posteriori.length > 1 ? posteriori[1] : posteriori[0];
        return probIsWinner;
    }

    public static Tuple toTuple(double[] data, String[] columnNames) {
        if (data == null) {
            return null;
        }

        List<StructField> fields = new ArrayList<>();
        for (int i = 0; i < columnNames.length; i++) {
            fields.add(new StructField(columnNames[i], DataTypes.DoubleType));
        }

        StructType schema = new StructType(fields);
        return Tuple.of(schema, data);
    }

    // min_deck_size – the number of cards in the smallest phase deck
    class SPFeatureMinDeckSize extends SPFeature {
        public SPFeatureMinDeckSize() {
            super("min_deck_size", "the number of cards in the smallest phase deck");
        }

        public Object getValue(SPState state) {
            int minDeckSize = Integer.MAX_VALUE;
            minDeckSize = Math.min(minDeckSize, state.workerDeck.size());
            minDeckSize = Math.min(minDeckSize, state.buildingDeck.size());
            minDeckSize = Math.min(minDeckSize, state.aristocratDeck.size());
            minDeckSize = Math.min(minDeckSize, state.tradingDeck.size());
            return minDeckSize;
        }
    }

    // round – current round number
    class SPFeatureRound extends SPFeature {
        public SPFeatureRound() {
            super("round", "current round number");
        }

        public Object getValue(SPState state) {
            int currPhase = (state.phase < 4) ? state.phase : (state.phase == 4) ? 2 : 0;
            return (double) state.round + (currPhase % 4) / 4.0;
        }
    }

    // points – current player points
    class SPFeaturePoints extends SPFeature {
        public SPFeaturePoints() {
            super("points", "current player points");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn];
        }
    }

    // points_diff – current player points relative to the opponent (assumes two
    // players)
    class SPFeaturePointsDiff extends SPFeature {
        public SPFeaturePointsDiff() {
            super("points_diff", "current player points relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn] - state.playerPoints[1 - state.playerTurn];
        }
    }

    // rubles – current player rubles (money)
    class SPFeatureRubles extends SPFeature {
        public SPFeatureRubles() {
            super("rubles", "current player rubles (money)");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn];
        }
    }

    // rubles_diff – current player rubles (money) relative to the opponent
    class SPFeatureRublesDiff extends SPFeature {
        public SPFeatureRublesDiff() {
            super("rubles_diff", "current player rubles (money) relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn] - state.playerRubles[1 - state.playerTurn];
        }
    }

    // points_round_gain – the number of points the current player is gaining per
    // round
    class SPFeaturePointsRoundGain extends SPFeature {
        public SPFeaturePointsRoundGain() {
            super("points_round_gain", "the number of points the current player is gaining per round");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound;
        }
    }

    // points_round_gain_diff – the number of points the current player is gaining
    // per round relative to the opponent
    class SPFeaturePointsRoundGainDiff extends SPFeature {
        public SPFeaturePointsRoundGainDiff() {
            super("points_round_gain_diff",
                    "the number of points the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            int opponentPointsPerRound = state.playerWorkers.get(1 - state.playerTurn).stream()
                    .mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound - opponentPointsPerRound;
        }
    }

    // rubles_round_gain – the number of rubles the current player is gaining per
    // round
    class SPFeatureRublesRoundGain extends SPFeature {
        public SPFeatureRublesRoundGain() {
            super("rubles_round_gain", "the number of rubles the current player is gaining per round");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            return rublesPerRound;
        }
    }

    // rubles_round_gain_diff – the number of rubles the current player is gaining
    // per round relative to the opponent
    class SPFeatureRublesRoundGainDiff extends SPFeature {
        public SPFeatureRublesRoundGainDiff() {
            super("rubles_round_gain_diff",
                    "the number of rubles the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            int opponentRublesPerRound = state.playerWorkers.get(1 - state.playerTurn).stream()
                    .mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            return rublesPerRound - opponentRublesPerRound;
        }
    }

    // unique_aristocrats – the number of unique aristocrats of the current player
    class SPFeatureUniqueAristocrats extends SPFeature {
        public SPFeatureUniqueAristocrats() {
            super("unique_aristocrats", "the number of unique aristocrats of the current player");
        }

        public Object getValue(SPState state) {
            return state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
        }
    }

    // unique_aristocrats_diff – the number of unique aristocrats of the current
    // player relative to the opponent
    class SPFeatureUniqueAristocratsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsDiff() {
            super("unique_aristocrats_diff",
                    "the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
            long opponentUniqueAristocrats = state.playerAristocrats.get(1 - state.playerTurn).stream().distinct()
                    .count();
            return uniqueAristocrats - opponentUniqueAristocrats;
        }
    }

    // unique_aristocrats_points_diff – the difference in potential point added
    // based on the number of unique aristocrats of the current player relative to
    // the opponent
    class SPFeatureUniqueAristocratsPointsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsPointsDiff() {
            super("unique_aristocrats_points_diff",
                    "the difference in potential point added based on the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = Math.min(SPState.MAX_UNIQUE_ARISTOCRATS,
                    state.playerAristocrats.get(state.playerTurn).stream().distinct().count());
            long opponentUniqueAristocrats = Math.min(SPState.MAX_UNIQUE_ARISTOCRATS,
                    state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count());
            return SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get((int) uniqueAristocrats)
                    - SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get((int) opponentUniqueAristocrats);
        }
    }

    // cards_in_hand – the number of cards in the current player hand
    class SPFeatureCardsInHand extends SPFeature {
        public SPFeatureCardsInHand() {
            super("cards_in_hand", "the number of cards in the current player hand");
        }

        public Object getValue(SPState state) {
            return state.playerHands.get(state.playerTurn).size();
        }
    }

    // cards_in_hand_diff – the number of cards in the current player hand relative
    // to the opponent
    class SPFeatureCardsInHandDiff extends SPFeature {
        public SPFeatureCardsInHandDiff() {
            super("cards_in_hand_diff", "the number of cards in the current player hand relative to the opponent");
        }

        public Object getValue(SPState state) {
            int cardsInHand = state.playerHands.get(state.playerTurn).size();
            int opponentCardsInHand = state.playerHands.get(1 - state.playerTurn).size();
            return cardsInHand - opponentCardsInHand;
        }
    }

    // hand_space_avl – the number of empty slots in a player's hand
    class SPFeatureHandSpace extends SPFeature {
        public SPFeatureHandSpace() {
            super("hand_space_avl", "the number of empty slots in a player's hand");
        }

        public Object getValue(SPState state) {
            boolean hasWarehouse = state.playerBuildings.get(state.playerTurn).stream()
                    .anyMatch(b -> b.name.equals("Warehouse"));
            return (hasWarehouse ? 4 : 3) - state.playerHands.get(state.playerTurn).size();
        }
    }

    // hand_space_avl_diff – the difference in number of empty slots in a player's
    // hand
    class SPFeatureHandSpaceDiff extends SPFeature {
        public SPFeatureHandSpaceDiff() {
            super("hand_space_avl_diff", "the number of empty slots in a player's hand");
        }

        public Object getValue(SPState state) {
            boolean curHasWarehouse = state.playerBuildings.get(state.playerTurn).stream()
                    .anyMatch(b -> b.name.equals("Warehouse"));
            int curHandSpace = (curHasWarehouse ? 4 : 3) - state.playerHands.get(state.playerTurn).size();
            boolean oppHasWarehouse = state.playerBuildings.get(1 - state.playerTurn).stream()
                    .anyMatch(b -> b.name.equals("Warehouse"));
            int oppHandSpace = (oppHasWarehouse ? 4 : 3) - state.playerHands.get(1 - state.playerTurn).size();
            return curHandSpace - oppHandSpace;
        }
    }

    // buying_adv - the advantage the current player has over the number of new card
    // he's able to acquired the following round
    class SPFeatureBuyingAdv extends SPFeature {
        public SPFeatureBuyingAdv() {
            super("buying_adv",
                    "the advantage the current player has over the number of new card he's able to acquired the following round");
        }

        public Object getValue(SPState state) {
            int numCardsInRows = state.upperCardRow.size() + state.lowerCardRow.size();
            int nextStartingPlayer = state.startingPlayer[(state.phase + 1) % SPState.NUM_DECKS];
            if (numCardsInRows % 2 == 0) {
                return 0;
            } else {
                if (nextStartingPlayer == 0) {
                    return 1;
                } else {
                    return -1;
                }
            }
        }
    }

    // total_cost_in_hand - the sum of the costs of all cards in the player's hand.
    // a high number suggests the hand is difficult to empty.
    class SPFeatureCostInHand extends SPFeature {
        public SPFeatureCostInHand() {
            super("total_cost_in_hand",
                    "the sum of the costs of all cards in the player's hand. a high number suggests the hand is difficult to empty.");
        }

        public Object getValue(SPState state) {
            int sum = 0;

            for (SPCard card : state.playerHands.get(state.playerTurn)) {
                sum += card.cost;
            }

            return sum;
        }
    }

    // buyable_cards_in_hand - the number of cards in hand that the player can
    // currently afford to play
    class SPFeatureBuyableCardsInHand extends SPFeature {
        public SPFeatureBuyableCardsInHand() {
            super("buyable_cards_in_hand", "the number of cards in hand that the player can currently afford to play");
        }

        public Object getValue(SPState state) {
            int count = 0;

            for (SPCard card : state.playerHands.get(state.playerTurn)) {
                if (state.playerRubles[state.playerTurn] > card.cost) {
                    count += 1;
                }
            }

            return count;
        }
    }

    // duplicate_aristocrat_count - the number of Aristocrat cards that are
    // duplicates of others in play. these are prime candidates for upgrades.
    class SPFeatureDupAristoCount extends SPFeature {
        public SPFeatureDupAristoCount() {
            super("duplicate_aristocrat_count",
                    "the number of Aristocrat cards that are duplicates of others in play. these are prime candidates for upgrades.");
        }

        public Object getValue(SPState state) {
            return state.playerAristocrats.get(state.playerTurn).size()
                    - state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
        }
    }

    // num_legal_moves - the number of legal moves available to the current player
    class SPFeatureNumLegalMoves extends SPFeature {
        public SPFeatureNumLegalMoves() {
            super("num_legal_moves", "the number of legal moves available to the current player.");
        }

        public Object getValue(SPState state) {
            return state.getLegalActions().size();
        }
    }

    // est_rounds_left - the estimate of rounds left in the game - CatBoost approach
    class SPFeatureRoundsLeft extends SPFeature {
        public SPFeatureRoundsLeft() {
            super("rounds_left", "the estimate of rounds left in the game - CatBoost approach");
        }

        public Object getValue(SPState state) {
            if (roundsLeftModel == null) {
                System.err.println("Rounds Left model not loaded. Returning fallback value -1.");
                return -1.0; // Return a default/fallback value
            }

            try {
                // 1. Get feature values from the existing helper method
                ArrayList<Object> values = getRLTrainFeatureValues(state);
                
                // 2. Convert features to float[] for CatBoost
                float[] numericFeatures = new float[values.size()];
                for (int i = 0; i < values.size(); i++) {
                    numericFeatures[i] = ((Number) values.get(i)).floatValue();
                }

                // 3. Predict
                CatBoostPredictions prediction = roundsLeftModel.predict(numericFeatures, new String[0]);
                
                // 4. Return the first (and only) prediction value
                return prediction.get(0, 0);

            } catch (Exception e) {
                System.err.println("[SPFeatureRoundsLeft] CatBoost prediction exception: " + e.getMessage());
                e.printStackTrace();
                return -1.0;
            }
        }
    }

    // Special features for rounds left estimator
    class SPFeatureCardsInDeck extends SPFeature {
        private int phase;

        public SPFeatureCardsInDeck(int phase) {
            super("cards_in_deck_" + phase, "number of cards remaining in phase " + phase + " deck");
            this.phase = phase;
        }

        public Object getValue(SPState state) {
            if (phase == 0) {
                return state.workerDeck.size();
            } else if (phase == 1) {
                return state.buildingDeck.size();
            } else if (phase == 2) {
                return state.aristocratDeck.size();
            } else {
                return state.tradingDeck.size();
            }
        }
    }  

    class SPFeaturePointsPerPhase extends SPFeature {
        private int phase;
        private int player;

        public SPFeaturePointsPerPhase(int phase, int player) {
            super("points_phase_" + phase + "_player_" + player, "points of player " + player + " in phase " + phase);
            this.phase = phase;
            this.player = player;
        }

        public Object getValue(SPState state) {
            ArrayList<SPCard> cards;
            if (phase == 0) {
                cards = state.playerWorkers.get(player);
            } else if (phase == 1) {
                cards = state.playerBuildings.get(player);
            } else {
                cards = state.playerAristocrats.get(player);
            }
            return cards.stream().mapToInt(card -> card.points).sum();
        }
    }

    class SPFeatureRublesPerPhase extends SPFeature {
        private int phase;
        private int player;

        public SPFeatureRublesPerPhase(int phase, int player) {
            super("rubles_phase_" + phase + "_player_" + player, "rubles of player " + player + " in phase " + phase);
            this.phase = phase;
            this.player = player;
        }

        public Object getValue(SPState state) {
            ArrayList<SPCard> cards;
            if (phase == 0) {
                cards = state.playerWorkers.get(player);
            } else if (phase == 1) {
                cards = state.playerBuildings.get(player);
            } else {
                cards = state.playerAristocrats.get(player);
            }
            return cards.stream().mapToInt(card -> card.rubles).sum();
        }
    }

    // player_#_rubles – rubles (money) of player #
    class SPFeaturePlayerNRubles extends SPFeature {
        private int playerIndex;

        public SPFeaturePlayerNRubles(int playerIndex) {
            super("player_" + playerIndex + "_rubles", "rubles (money) of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            return state.playerRubles[playerIndex];
        }
    }

    // player_#_points – points of player #
    class SPFeaturePlayerNPoints extends SPFeature {
        private int playerIndex;

        public SPFeaturePlayerNPoints(int playerIndex) {
            super("player_" + playerIndex + "_points", "points of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            return state.playerPoints[playerIndex];
        }
    }

    class SPFeatureUniqueAristocratPoints extends SPFeature {
        private int playerIndex;

        public SPFeatureUniqueAristocratPoints(int playerIndex) {
            super("player_" + playerIndex + "_unique_aristocrat_points", "points of unique aristocrats of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            int uniqueAristocrats = state.playerAristocrats.get(playerIndex).stream().distinct().mapToInt(card -> card.points).sum();
            return (uniqueAristocrats * (uniqueAristocrats + 1) / 2);
        }
    }

    class SPFeatureDuplicateAristocrats extends SPFeature {
        private int playerIndex;

        public SPFeatureDuplicateAristocrats(int playerIndex) {
            super("player_" + playerIndex + "_duplicate_aristocrats", "number of duplicate aristocrats of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            Set<SPCard> uniqueAristocrats = new HashSet<>(state.playerAristocrats.get(playerIndex));
            return state.playerAristocrats.get(playerIndex).size() - uniqueAristocrats.size();
        }
    }

    // current_round - current round number
    class SPFeatureCurrentRound extends SPFeature {
        public SPFeatureCurrentRound() {
            super("current_round", "current round number");
        }

        public Object getValue(SPState state) {
            int currPhase = (state.phase < 4) ? state.phase : (state.phase == 4) ? 2 : 0;
            double round = (double) state.round + (currPhase % 4) / 4.0;
            return round;
        }
    }

    class SPFeatureCardsInHandN extends SPFeature {
        private int playerIndex;

        public SPFeatureCardsInHandN(int playerIndex) {
            super("cards_in_hand_" + playerIndex, "number of cards in hand of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            return state.playerHands.get(playerIndex).size();
        }
    }

    class SPFeatureCardsOfferedTop extends SPFeature {
        public SPFeatureCardsOfferedTop() {
            super("cards_offered_top", "number of cards offered on the top of the deck");
        }

        public Object getValue(SPState state) {
            return state.upperCardRow.size();
        }
    }

    // cards_offered_bottom – number of cards offered on the bottom of the deck
    class SPFeatureCardsOfferedBottom extends SPFeature {
        public SPFeatureCardsOfferedBottom() {
            super("cards_offered_bottom", "number of cards offered on the bottom of the deck");
        }

        public Object getValue(SPState state) {
            return state.lowerCardRow.size();
        }
    }

    public static void main(String[] args) {
        new AIDanSPStateFeaturesRF1();
    }

}