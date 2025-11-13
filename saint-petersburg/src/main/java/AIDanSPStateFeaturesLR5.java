import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;      // Added
import java.util.ArrayList;     // Added
import java.util.List;

import smile.classification.LogisticRegression;

/**
 * Implements a heuristic for SPState using multiple Logistic Regression models,
 * one for each "phase" of the game (Early, Mid, Late).
 * The game phase is determined by the `min_deck_size` feature.
 * * This class will:
 * 1. Load three separate models (Early, Mid, Late) from disk.
 * 2. If models are not found, it will read the *entire* CSV file.
 * 3. It will "bin" the training data based on `min_deck_size` cutoffs.
 * 4. It will train one model for each bin and save them to disk.
 * 5. When predict() is called, it will check the state's `min_deck_size` and
 * use the appropriate model to return a heuristic score.
 */
public class AIDanSPStateFeaturesLR5 {
    
    // --- Model Configuration ---
    
    // Define your cutoffs here based on min_deck_size
    // Early Game = min_deck_size > CUTOFF_MID_GAME
    // Mid Game   = min_deck_size > CUTOFF_LATE_GAME (and <= CUTOFF_MID_GAME)
    // Late Game  = min_deck_size <= CUTOFF_LATE_GAME
    static final int CUTOFF_MID_GAME = 15;
    static final int CUTOFF_LATE_GAME = 10;

    // Define filenames for each model
    String modelFilename_Early = "AIDanSPLogisticRegression_Early.model";
    String modelFilename_Mid = "AIDanSPLogisticRegression_Mid.model";
    String modelFilename_Late = "AIDanSPLogisticRegression_Late.model";

    // Create three model instances
    LogisticRegression.Binomial model_Early;
    LogisticRegression.Binomial model_Mid;
    LogisticRegression.Binomial model_Late;

    // --- Feature Configuration ---
    ArrayList<SPFeature> features;
    
    // Cache the minDeckFeature for fast lookups in predict()
    private SPFeatureMinDeckSize minDeckFeature = new SPFeatureMinDeckSize();
    
    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public AIDanSPStateFeaturesLR5() {
    	features = new ArrayList<>();

        features = new ArrayList<>();
        // features.add(new SPFeatureMinDeckSize());
        // features.add(new SPFeaturePoints());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRubles());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureUniqueAristocratsPointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsPointsDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeaturePointsRoundGain());
        // features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureRublesRoundGain());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureCardsInHand());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureCardsInHandDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureHandSpaceDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureHandSpaceDiff(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureBuyableCardsInHand());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureBuyableCardsInHand(), new SPFeatureMinDeckSize()));
        // features.add(new SPFeatureDupAristoCount());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureDupAristoCount(), new SPFeatureMinDeckSize()));

        initializeModel();
    }

    /**
     * Initializes all three game-phase models.
     * If any model file is missing, triggers the learning process for all three.
     */
    private void initializeModel() {
        // Check if all three models exist
        boolean allModelsExist = java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename_Early)) &&
                                 java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename_Mid)) &&
                                 java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename_Late));

        if (!allModelsExist) {
            System.out.println("One or more game-phase models not found. Generating all models...");
            learnModels(); // Renamed from learnModel()
        }

        // Try to load all three models
        try {
            System.out.println("Loading Early-Game Model: " + modelFilename_Early);
            try (ObjectInputStream ois = new ObjectInputStream(new java.io.FileInputStream(modelFilename_Early))) {
                model_Early = (LogisticRegression.Binomial) ois.readObject();
            }
            
            System.out.println("Loading Mid-Game Model: " + modelFilename_Mid);
            try (ObjectInputStream ois = new ObjectInputStream(new java.io.FileInputStream(modelFilename_Mid))) {
                model_Mid = (LogisticRegression.Binomial) ois.readObject();
            }

            System.out.println("Loading Late-Game Model: " + modelFilename_Late);
            try (ObjectInputStream ois = new ObjectInputStream(new java.io.FileInputStream(modelFilename_Late))) {
                model_Late = (LogisticRegression.Binomial) ois.readObject();
            }
            
            System.out.println("All game-phase models loaded successfully.");

        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading one of the models. You may need to delete all .model files and retrain.");
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

    public String getCSVRow(SPState state, boolean[] isWinner) {
        int currentPlayerIndex = state.playerTurn;
        int winnerVal = isWinner[currentPlayerIndex] ? 1 : 0;
        StringBuilder row = new StringBuilder();
        for (SPFeature feature : features) {
            row.append(feature.getValue(state)).append(",");
        }
        row.append(winnerVal);
        return row.toString();
    }

    public String getCSVRows(SPGameTranscript transcript) {
        StringBuilder rows = new StringBuilder();
        boolean[] isWinner = transcript.getWinners();
        for (SPState state : transcript.getStates()) {
            rows.append(getCSVRow(state, isWinner)).append("\n");
        }
        return rows.toString();
    }

    public void generateCSVData(String filename, int numGames) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println(getCSVHeader());
            for (int i = 0; i < numGames; i++) {
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
                writer.print(getCSVRows(transcript));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<String> getFeatureNames() {
        ArrayList<String> names = new ArrayList<>();
        for (SPFeature feature : features) {
            names.add(feature.getName());
        }
        return names;
    }

    /**
     * Main learning method. Reads the CSV and filters data into three bins
     * based on min_deck_size, then trains a model for each bin.
     */
    public void learnModels() {
        String trainingDataFile = "AIDanSPTrainingDataFlatMCvsFlatMC.csv";

        ArrayList<String> desiredFeatureNames = getFeatureNames();
        List<Integer> featureIndicesToKeep = new ArrayList<>();

        // Create lists for each game phase
        List<double[]> values_Early = new ArrayList<>();
        List<Integer> labels_Early = new ArrayList<>();
        List<double[]> values_Mid = new ArrayList<>();
        List<Integer> labels_Mid = new ArrayList<>();
        List<double[]> values_Late = new ArrayList<>();
        List<Integer> labels_Late = new ArrayList<>();

        int minDeckColumnIndex = -1;

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            
            // --- 1. Process Header ---
            String headerLine = br.readLine(); 
            String[] allCsvHeaders = headerLine.split(",");

            for (int i = 0; i < allCsvHeaders.length; i++) {
                String csvColumnName = allCsvHeaders[i].trim(); // Use trim() for safety
                if (desiredFeatureNames.contains(csvColumnName)) {
                    featureIndicesToKeep.add(i);
                }
                // Find the min_deck_size column
                if (csvColumnName.equals("min_deck_size")) {
                    minDeckColumnIndex = i;
                }
            }
            
            if (minDeckColumnIndex == -1) {
                System.err.println("CRITICAL ERROR: 'min_deck_size' column not found in " + trainingDataFile);
                return;
            }
            if (featureIndicesToKeep.size() != desiredFeatureNames.size()) {
                System.err.println("ERROR: Mismatch between features in code and CSV!");
                System.err.println("Code features: " + desiredFeatureNames);
                System.err.println("Found " + featureIndicesToKeep.size() + " matching columns in CSV: " + featureIndicesToKeep);
                return;
            }

            // --- 2. Process Data Rows ---
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < allCsvHeaders.length) continue; // Skip malformed lines
                
                // Get the minDeck value for this row
                int minDeck = (int) Double.parseDouble(parts[minDeckColumnIndex]); // Use parseDouble for safety

                // Build the feature row
                double[] row = new double[featureIndicesToKeep.size()];
                for (int i = 0; i < featureIndicesToKeep.size(); i++) {
                    int csvIndex = featureIndicesToKeep.get(i);
                    row[i] = Double.parseDouble(parts[csvIndex]);
                }
                
                // Get the label
                int label = Integer.parseInt(parts[parts.length - 1]);

                // --- This is the new logic: Filter data into bins ---
                if (minDeck > CUTOFF_MID_GAME) {
                    values_Early.add(row);
                    labels_Early.add(label);
                } else if (minDeck > CUTOFF_LATE_GAME) {
                    values_Mid.add(row);
                    labels_Mid.add(label);
                } else {
                    values_Late.add(row);
                    labels_Late.add(label);
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        System.out.println("Data read complete. Bins:");
        System.out.println(" - Early Game (> " + CUTOFF_MID_GAME + "): " + values_Early.size() + " samples");
        System.out.println(" - Mid Game   (> " + CUTOFF_LATE_GAME + "): " + values_Mid.size() + " samples");
        System.out.println(" - Late Game  (<= " + CUTOFF_LATE_GAME + "): " + values_Late.size() + " samples");

        // --- 3. Train and Save Models ---
        System.out.println("Training Early-Game Model...");
        trainAndSave(values_Early, labels_Early, modelFilename_Early);
        
        System.out.println("Training Mid-Game Model...");
        trainAndSave(values_Mid, labels_Mid, modelFilename_Mid);

        System.out.println("Training Late-Game Model...");
        trainAndSave(values_Late, labels_Late, modelFilename_Late);
        
        System.out.println("All models trained and saved.");
    }

    /**
     * Helper method to train a single model and save it to a file.
     */
    private void trainAndSave(List<double[]> values, List<Integer> labels, String filename) {
        if (values.isEmpty()) {
            System.err.println("WARNING: No data for model " + filename + ". Skipping training.");
            return;
        }

        double[][] X = values.toArray(new double[0][]);
        int[] y = labels.stream().mapToInt(i -> i).toArray();

        // Perform logistic regression
        LogisticRegression.Binomial model = LogisticRegression.binomial(X, y);

        // Save the model to its specific file
        try (ObjectOutputStream oos = new ObjectOutputStream(new java.io.FileOutputStream(filename))) {
            oos.writeObject(model);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Model saved to " + filename);
        // Optional: Print coefficients for this specific model
        System.out.println("Coefficients for " + filename + ":");
        System.out.printf("%.4f\tIntercept%n", model.coefficients()[0]);
        for (int i = 0; i < model.coefficients().length - 1; i++) {
            System.out.printf("%.4f\t%s%n", model.coefficients()[i + 1], features.get(i).getName());
        }
    }


    /**
     * Predicts the win probability for the current player in the given state.
     * This method automatically selects the correct model (Early, Mid, Late)
     * based on the state's `min_deck_size`.
     * * @param state The current SPState.
     * @return A heuristic score (probability of winning) between 0.0 and 1.0.
     */
    public double predict(SPState state) {
        // --- 1. Get the min_deck_size ---
        // We use the cached feature object for this
        int minDeck = (Integer) minDeckFeature.getValue(state);

        // --- 2. Get the feature vector (same as before) ---
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }

        // --- 3. Select the correct model based on cutoffs ---
        if (minDeck > CUTOFF_MID_GAME) {
            // Use Early-Game Model
            if (model_Early != null) {
                return model_Early.score(featureValues);
            }
        } else if (minDeck > CUTOFF_LATE_GAME) {
            // Use Mid-Game Model
            if (model_Mid != null) {
                return model_Mid.score(featureValues);
            }
        } else {
            // Use Late-Game Model
            if (model_Late != null) {
                return model_Late.score(featureValues);
            }
        }
        
        // Fallback in case a model is null or fails
        System.err.println("WARNING: Appropriate model not found or not loaded for minDeck=" + minDeck + ". Returning default 0.5 score.");
        return 0.5;
    }

    // --- All Feature Definitions Below (Unchanged) ---

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
            return state.round;
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

    // points_diff – current player points relative to the opponent (assumes two players)
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

    // points_round_gain – the number of points the current player is gaining per round
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

    // points_round_gain_diff – the number of points the current player is gaining per round relative to the opponent
    class SPFeaturePointsRoundGainDiff extends SPFeature {
        public SPFeaturePointsRoundGainDiff() {
            super("points_round_gain_diff", "the number of points the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            int opponentPointsPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound - opponentPointsPerRound;
        }
    }

    // rubles_round_gain – the number of rubles the current player is gaining per round
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

    // rubles_round_gain_diff – the number of rubles the current player is gaining per round relative to the opponent
    class SPFeatureRublesRoundGainDiff extends SPFeature {
        public SPFeatureRublesRoundGainDiff() {
            super("rubles_round_gain_diff", "the number of rubles the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            int opponentRublesPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
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

    // unique_aristocrats_diff – the number of unique aristocrats of the current player relative to the opponent
    class SPFeatureUniqueAristocratsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsDiff() {
            super("unique_aristocrats_diff", "the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
            long opponentUniqueAristocrats = state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count();
            return uniqueAristocrats - opponentUniqueAristocrats;
        }
    }
    
    // unique_aristocrats_points_diff – the difference in potential point added based on the number of unique aristocrats of the current player relative to the opponent
    class SPFeatureUniqueAristocratsPointsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsPointsDiff() {
            super("unique_aristocrats_points_diff", "the difference in potential point added based on the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = Math.min(SPState.MAX_UNIQUE_ARISTOCRATS, state.playerAristocrats.get(state.playerTurn).stream().distinct().count());
            long opponentUniqueAristocrats = Math.min(SPState.MAX_UNIQUE_ARISTOCRATS, state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count());
            return SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get((int) uniqueAristocrats) - SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get((int) opponentUniqueAristocrats);
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

    // cards_in_hand_diff – the number of cards in the current player hand relative to the opponent
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
        	boolean hasWarehouse = state.playerBuildings.get(state.playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
    		return (hasWarehouse ? 4 : 3) - state.playerHands.get(state.playerTurn).size();
        }
    }
    
    // hand_space_avl_diff – the difference in number of empty slots in a player's hand
    class SPFeatureHandSpaceDiff extends SPFeature {
        public SPFeatureHandSpaceDiff() {
            super("hand_space_avl_diff", "the number of empty slots in a player's hand");
        }

        public Object getValue(SPState state) {
        	boolean curHasWarehouse = state.playerBuildings.get(state.playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
    		int curHandSpace = (curHasWarehouse ? 4 : 3) - state.playerHands.get(state.playerTurn).size();
    		boolean oppHasWarehouse = state.playerBuildings.get(1 - state.playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
    		int oppHandSpace = (oppHasWarehouse ? 4 : 3) - state.playerHands.get(1 - state.playerTurn).size();
    		return curHandSpace - oppHandSpace;
        }
    }
    
    // buying_adv - the advantage the current player has over the number of new card he's able to acquired the following round
    class SPFeatureBuyingAdv extends SPFeature {
    	public SPFeatureBuyingAdv() {
    		super("buying_adv", "the advantage the current player has over the number of new card he's able to acquired the following round");
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
    
    // total_cost_in_hand - the sum of the costs of all cards in the player's hand. a high number suggests the hand is difficult to empty.
    class SPFeatureCostInHand extends SPFeature {
    	public SPFeatureCostInHand() {
    		super("total_cost_in_hand", "the sum of the costs of all cards in the player's hand. a high number suggests the hand is difficult to empty.");
    	}
    	
    	public Object getValue(SPState state) {
    		int sum = 0;
    		
    		for (SPCard card : state.playerHands.get(state.playerTurn)) {
    			sum += card.cost;
    		}
    		
    		return sum;
    	}
    }
    
    // buyable_cards_in_hand - the number of cards in hand that the player can currently afford to play
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
    
    // duplicate_aristocrat_count - the number of Aristocrat cards that are duplicates of others in play. these are prime candidates for upgrades.
    class SPFeatureDupAristoCount extends SPFeature {
    	public SPFeatureDupAristoCount() {
    		super("duplicate_aristocrat_count", "the number of Aristocrat cards that are duplicates of others in play. these are prime candidates for upgrades.");
    	}
    	
    	public Object getValue(SPState state) {
    		return state.playerAristocrats.get(state.playerTurn).size() - state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
    	}
    }

    public static void main(String[] args) {
        System.out.println("Initializing Staged-Model Features (LR3)...");
        new AIDanSPStateFeaturesLR5();
        System.out.println("Initialization complete.");
    }

}