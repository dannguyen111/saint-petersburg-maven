import java.io.BufferedReader;
import java.io.File; // Added
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet; // Added
import java.util.List;
import java.util.Set; // Added

import ai.catboost.CatBoostError; // Added
import ai.catboost.CatBoostModel; // Added
import ai.catboost.CatBoostPredictions;
import smile.classification.LogisticRegression;

public class AIDanSPStateFeaturesLR4 {
    String modelFilename = "AIDanSPLogisticRegression4.model";
    LogisticRegression.Binomial model;
    ArrayList<SPFeature> features;
    
    // --- Copied from RF1 ---
    ArrayList<SPFeature> rlFeatures;
    private CatBoostModel roundsLeftModel;
    String rlModelFilename = "rounds_left_model.cbm";
    // --- End Copy ---

    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }
    
    // --- Copied from RF1 ---
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
    // --- End Copy ---

    public AIDanSPStateFeaturesLR4() {
    	features = new ArrayList<>();
        // features.add(new SPFeatureRoundsLeft());
        // features.add(new SPFeaturePoints());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureRoundsLeft()));
        features.add(new SPFeaturePointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureRoundsLeft()));
        // features.add(new SPFeatureRubles());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureRublesDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureUniqueAristocratsPointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsPointsDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeaturePointsRoundGain());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureRoundsLeft()));
        features.add(new SPFeaturePointsRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureRoundsLeft()));
        // features.add(new SPFeatureRublesRoundGain());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureRublesRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureRoundsLeft()));
        // features.add(new SPFeatureCardsInHand());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureRoundsLeft()));
        // features.add(new SPFeatureNumLegalMoves());
        // features.add(new SPFeatureInteractionTerm(new SPFeatureNumLegalMoves(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureCardsInHandDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureRoundsLeft()));
        features.add(new SPFeatureHandSpaceDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureHandSpaceDiff(), new SPFeatureRoundsLeft()));
        
        // --- Copied from RF1 ---
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

        initializeRoundsLeftModel();
        // --- End Copy ---

        initializeModel();
    }
    
    // --- Copied from RF1 ---
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
    // --- End Copy ---

    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
        }
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
            model = (LogisticRegression.Binomial) ois.readObject();
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

    public String getCSVRow(SPState state, boolean[] isWinner) {
        // This function might need updating if you want to use *actual*
        // rounds_left for training, like RF1 does.
        // For now, it uses the *estimated* rounds_left.
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

    public void learnModel() {
        String trainingDataFile = "AIDanSPTrainingDataFlatMCvsFlatMCRL.csv";

        ArrayList<String> desiredFeatureNames = getFeatureNames();
        List<Integer> featureIndicesToKeep = new ArrayList<>();

        List<double[]> values = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            
            String headerLine = br.readLine(); 
            String[] allCsvHeaders = headerLine.split(",");

            for (int i = 0; i < allCsvHeaders.length; i++) {
                String csvColumnName = allCsvHeaders[i];
                if (desiredFeatureNames.contains(csvColumnName)) {
                    featureIndicesToKeep.add(i);
                }
            }
            
            if (featureIndicesToKeep.size() != desiredFeatureNames.size()) {
                System.err.println("ERROR: Mismatch between features in code and CSV!");
                System.err.println("Code features: " + desiredFeatureNames);
                System.err.println("Found " + featureIndicesToKeep.size() + " matching columns in CSV.");
                return;
            }

            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                
                // Build the feature row *selectively* using our index map
                double[] row = new double[featureIndicesToKeep.size()];
                for (int i = 0; i < featureIndicesToKeep.size(); i++) {
                    int csvIndex = featureIndicesToKeep.get(i);
                    row[i] = Double.parseDouble(parts[csvIndex]);
                }
                values.add(row);
                
                // Add the label (it's always the last part)
                labels.add(Integer.parseInt(parts[parts.length - 1]));
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[][] X = values.toArray(new double[0][]);
        int[] y = labels.stream().mapToInt(i -> i).toArray();

        // Perform logistic regression using the Smile library
        LogisticRegression.Binomial model = LogisticRegression.binomial(X, y);

        // Save the model to a file using an ObjectOutputStream
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelFilename))) {
            oos.writeObject(model);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Print the model coefficients along with their feature names
        // This printout will now work correctly!
        System.out.println("Model coefficients:");
        System.out.println(features.size() + " features"); // Will print 14
        System.out.println(model.coefficients().length + " coefficients"); // Will print 15 (14 + intercept)
        System.out.printf("%.4f\tIntercept%n", model.coefficients()[0]);
        for (int i = 0; i < model.coefficients().length - 1; i++) { // Will loop 14 times
            // This is now safe and won't crash
            System.out.printf("%.4f\t%s%n", model.coefficients()[i + 1], features.get(i).getName());
        }
    }

    public double predict(SPState state) {
        // Create a double array for the feature values
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }

        // Use the logistic regression model to predict the probability of winning
        return model.score(featureValues);
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

        // This method is now identical to RF1's and will work.
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
    
    // --- Copied from RF1: Inner classes for RL features ---
    
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
    // --- End Copy ---


    public static void main(String[] args) {
        new AIDanSPStateFeaturesLR4(); // Fixed typo
    }

}